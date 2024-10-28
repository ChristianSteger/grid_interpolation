! Description: Auxiliary functions to remap data from unstructured grid to
!							 to cell centres of equally spaced (dx = dy) regular grid
!							 via inverse distance weighted interpolation
!
!	Author: Christian R. Steger, October 2024

MODULE idw_interp_esrg

	IMPLICIT NONE
	PRIVATE
	PUBLIC :: assign_points_to_cells, nearest_neighbours_esrg

	TYPE :: point_attr
		INTEGER :: index
		REAL :: dist_sq
	END TYPE point_attr

	CONTAINS

	! ---------------------------------------------------------------------------
	! Assign points to equally spaced regular grid (esrg)
	! ---------------------------------------------------------------------------

  SUBROUTINE assign_points_to_cells(points, x_axis, y_axis, grid_spac, &
  	index_of_pts, indptr, num_ppgc)

    REAL, DIMENSION(:, :), INTENT(IN) :: points ! (# of pts, 2)
    REAL, DIMENSION(:), INTENT(IN) :: x_axis, y_axis
    REAL, INTENT(IN) :: grid_spac
    INTEGER, DIMENSION(:), INTENT(OUT) :: index_of_pts ! (num_points)
    INTEGER, DIMENSION(:, :), INTENT(OUT) :: indptr ! (len_y, len_x)
    INTEGER, DIMENSION(:, :), INTENT(OUT) :: num_ppgc ! (len_y, len_x)

    INTEGER :: ind_x, ind_y
    INTEGER :: i, cumsum
    INTEGER :: len_x, len_y

    ! Shape of regular grid
		len_x = SIZE(x_axis)
		len_y = SIZE(y_axis)

    ! Compute number of points in each cell
    num_ppgc = 0 ! ppgc: points per grid cell
    DO i = 1, SIZE(points, 1)
      ind_x = INT((points(i, 1) - x_axis(1) + grid_spac / 2.0) / grid_spac) + 1
      IF (ind_x < 1) THEN
      	ind_x = 1
      END IF
      IF (ind_x > len_x) THEN
      	ind_x = len_x
			END IF
      ind_y = INT((points(i, 2) - y_axis(1) + grid_spac / 2.0) / grid_spac) + 1
      IF (ind_y < 1) THEN
      	ind_y = 1
      END IF
      IF (ind_y > len_y) THEN
      	ind_y = len_y
      END IF
      num_ppgc(ind_y, ind_x) = num_ppgc(ind_y, ind_x) + 1
    END DO

    ! Compute index pointer array
    cumsum = 1
    DO ind_y = 1, len_y
      DO ind_x = 1, len_x
        indptr(ind_y, ind_x) = cumsum
        cumsum = cumsum + num_ppgc(ind_y, ind_x)
      END DO
    END DO

    ! Assign index of points
    num_ppgc = 0
    DO i = 1, SIZE(points, 1)
      ind_x = INT((points(i, 1) - x_axis(1) + grid_spac / 2.0) / grid_spac) + 1
      IF (ind_x < 1) THEN
      	ind_x = 1
      END IF
      IF (ind_x > len_x) THEN
      	ind_x = len_x
			END IF
      ind_y = INT((points(i, 2) - y_axis(1) + grid_spac / 2.0) / grid_spac) + 1
      IF (ind_y < 1) THEN
      	ind_y = 1
      END IF
      IF (ind_y > len_y) THEN
      	ind_y = len_y
      END IF
      index_of_pts(indptr(ind_y, ind_x) + num_ppgc(ind_y, ind_x)) = i
      num_ppgc(ind_y, ind_x) = num_ppgc(ind_y, ind_x) + 1
    END DO

  END SUBROUTINE assign_points_to_cells

	! ---------------------------------------------------------------------------
	! Find nearest neighbours from grid cell centre and their distances
	! ---------------------------------------------------------------------------

	SUBROUTINE nearest_neighbours_esrg(points, &
		index_of_pts, indptr, num_ppgc, num_nn, &
		x_axis, y_axis, grid_spac, ind_x, ind_y, index_nn, dist_nn)

		REAL, DIMENSION(:, :), INTENT(IN) :: points ! (# of pts, 2)
		INTEGER, DIMENSION(:), INTENT(IN) :: index_of_pts ! (num_points)
		INTEGER, DIMENSION(:, :), INTENT(IN) :: indptr ! (len_y, len_x)
		INTEGER, DIMENSION(:, :), INTENT(IN) :: num_ppgc ! (len_y, len_x)
		INTEGER, INTENT(IN) :: num_nn
		REAL, DIMENSION(:), INTENT(IN) :: x_axis, y_axis
		REAL, INTENT(IN) :: grid_spac
		INTEGER, INTENT(IN) :: ind_x, ind_y
		INTEGER, DIMENSION(:), INTENT(OUT) :: index_nn
		REAL, DIMENSION(:), INTENT(OUT) :: dist_nn

		INTEGER :: len_x, len_y
		REAL :: radius_sq, dist_sq
		REAL, DIMENSION(:) :: dist_sq_nn(num_nn)
		INTEGER :: i, j, k
		INTEGER :: ind_x_nb, ind_y_nb
		INTEGER :: ind, ind_is, level, num_nn_found
		REAL, DIMENSION(:) :: centre(2)
		TYPE(point_attr), DIMENSION(100) :: points_cons
		INTEGER :: num_pts_cons
		TYPE(point_attr), DIMENSION(100) :: points_cons_next
		INTEGER :: num_pts_cons_next

		! Shape of regular grid
		len_x = SIZE(x_axis)
		len_y = SIZE(y_axis)

		! Initialise arrays
		index_nn = -1
		dist_sq_nn = HUGE(1.0)

		! Centre coordinates of the grid cell
		centre(1) = x_axis(ind_x)
		centre(2) = y_axis(ind_y)

		! -------------------------------------------------------------------------
		! Process centre grid cell
		! -------------------------------------------------------------------------

		level = 0
		radius_sq = (grid_spac * (level + 0.5)) ** 2

		! Assign points to list
		num_pts_cons = 0
		DO i = 1, num_ppgc(ind_y, ind_x)
			ind = index_of_pts(indptr(ind_y, ind_x) + (i - 1))
			dist_sq = (centre(1) - points(ind, 1)) ** 2 &
							+ (centre(2) - points(ind, 2)) ** 2
			num_pts_cons = num_pts_cons + 1
			points_cons(num_pts_cons)%index = ind
			points_cons(num_pts_cons)%dist_sq = dist_sq
		END DO

		! Potentially add new points
		num_pts_cons_next = 0
		DO i = 1, num_pts_cons
			ind = points_cons(i)%index
			dist_sq = points_cons(i)%dist_sq
			IF (dist_sq > radius_sq) THEN
				num_pts_cons_next = num_pts_cons_next + 1
				points_cons_next(num_pts_cons_next)%index = ind
				points_cons_next(num_pts_cons_next)%dist_sq = dist_sq
			ELSE IF (dist_sq < dist_sq_nn(num_nn)) THEN
				ind_is = count(dist_sq_nn < dist_sq) + 1 ! insertion index
				dist_sq_nn(ind_is + 1:num_nn) = dist_sq_nn(ind_is:num_nn - 1)
				dist_sq_nn(ind_is) = dist_sq
				index_nn(ind_is + 1:num_nn) = index_nn(ind_is:num_nn - 1)
				index_nn(ind_is) = ind
			END IF
		END DO

		num_nn_found = count(index_nn /= -1)

		! -------------------------------------------------------------------------
		! Process grid cells of surrounding 'frame(s)'
		! -------------------------------------------------------------------------

		DO WHILE (num_nn_found /= num_nn)

			level = level + 1
			radius_sq = (grid_spac * (level + 0.5)) ** 2

			! Assign points from bottom and top cells to list
			points_cons(:num_pts_cons_next) = points_cons_next(:num_pts_cons_next)
			num_pts_cons = num_pts_cons_next
			DO i = -level, level, 2 * level ! y-axis
				DO j = -level, level ! x-axis
					ind_y_nb = ind_y + i
					ind_x_nb = ind_x + j
					IF ((ind_x_nb < 1) .OR. (ind_x_nb > len_x) .OR. &
							(ind_y_nb < 1) .OR. (ind_y_nb > len_y)) THEN
						CYCLE
					END IF
					DO k = 1, num_ppgc(ind_y_nb, ind_x_nb)
						ind = index_of_pts(indptr(ind_y_nb, ind_x_nb) + (k - 1))
						dist_sq = (centre(1) - points(ind, 1)) ** 2 &
										+ (centre(2) - points(ind, 2)) ** 2
						num_pts_cons = num_pts_cons + 1
						points_cons(num_pts_cons)%index = ind
						points_cons(num_pts_cons)%dist_sq = dist_sq
					END DO
				END DO
			END DO

			! Assign points from left and right cells to list
			DO j = -level, level, 2 * level ! x-axis
				DO i = -level + 1, level - 1 ! y_axis
					ind_y_nb = ind_y + i
					ind_x_nb = ind_x + j
					IF ((ind_x_nb < 1) .OR. (ind_x_nb > len_x) .OR. &
							(ind_y_nb < 1) .OR. (ind_y_nb > len_y)) THEN
						CYCLE
					END IF
					DO k = 1, num_ppgc(ind_y_nb, ind_x_nb)
						ind = index_of_pts(indptr(ind_y_nb, ind_x_nb) + (k - 1))
						dist_sq = (centre(1) - points(ind, 1)) ** 2 &
										+ (centre(2) - points(ind, 2)) ** 2
						num_pts_cons = num_pts_cons + 1
						points_cons(num_pts_cons)%index = ind
						points_cons(num_pts_cons)%dist_sq = dist_sq          
          END DO
        END DO
      END DO

			! Potentially add new points
			num_pts_cons_next = 0
			DO i = 1, num_pts_cons
				ind = points_cons(i)%index
				dist_sq = points_cons(i)%dist_sq
				IF (dist_sq > radius_sq) THEN
					num_pts_cons_next = num_pts_cons_next + 1
					points_cons_next(num_pts_cons_next)%index = ind
					points_cons_next(num_pts_cons_next)%dist_sq = dist_sq
				ELSE IF (dist_sq < dist_sq_nn(num_nn)) THEN
					ind_is = count(dist_sq_nn < dist_sq) + 1 ! insertion index
					dist_sq_nn(ind_is + 1:num_nn) = dist_sq_nn(ind_is:num_nn - 1)
					dist_sq_nn(ind_is) = dist_sq
					index_nn(ind_is + 1:num_nn) = index_nn(ind_is:num_nn - 1)
					index_nn(ind_is) = ind
				END IF
			END DO

			num_nn_found = count(index_nn /= -1)

		END DO

		dist_nn = SQRT(dist_sq_nn)
    
	END SUBROUTINE nearest_neighbours_esrg

END MODULE idw_interp_esrg
