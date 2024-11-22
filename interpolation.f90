! Description: Functions to interpolate between regular and unstructured grid
!              (e.g. triangle mesh)
!
! Author: Christian R. Steger, October 2024

MODULE interpolation

  USE kd_tree
  USE query_esrg
  USE triangle_walk
  USE OMP_LIB
  IMPLICIT NONE

  CONTAINS

  ! ---------------------------------------------------------------------------
  ! Bilinear interpolation (regular -> unstructured grid)
  ! ---------------------------------------------------------------------------

  SUBROUTINE bilinear(x_axis, len_x, y_axis, len_y, data_in, &
    points, num_points, data_ip)

    INTEGER :: len_x, len_y, num_points
    REAL, DIMENSION(len_x) :: x_axis
    REAL, DIMENSION(len_y) :: y_axis
    REAL, DIMENSION(len_y, len_x) :: data_in
    REAL, DIMENSION(num_points, 2) :: points ! (# of pts, 2)
    REAL, DIMENSION(num_points) :: data_ip

    !f2py threadsafe
    !f2py intent(in) x_axis
    !f2py intent(hide) len_x
    !f2py intent(in) y_axis
    !f2py intent(hide) len_y
    !f2py intent(in) data_in
    !f2py intent(in) points
    !f2py intent(hide) num_points
    !f2py intent(out) data_ip

    INTEGER :: ind
    REAL :: delta
    REAL :: dx_inv, dy_inv
    INTEGER :: ind_x, ind_y
    REAL :: value_ofb = -9999.0  ! Out-of-bounds value
    REAL :: w_11, w_12, w_21, w_22
    REAL(8) :: time_1, time_2

    ! Compute inverse of x- and y-grid spacing
    delta = 0.0
    DO ind = 1, (len_x - 1)
      delta = delta + (x_axis(ind + 1) - x_axis(ind))
    END DO
    dx_inv = 1.0 / (delta / REAL(len_x - 1))
    delta = 0.0
    DO ind = 1, (len_y - 1)
      delta = delta + (y_axis(ind + 1) - y_axis(ind))
    END DO
    dy_inv = 1.0 / (delta / REAL(len_y - 1))

    ! Perform interpolation
    time_1 = OMP_GET_WTIME()
    !$OMP PARALLEL DO PRIVATE(ind_x, ind_y, w_11, w_12, w_21, w_22)
    DO ind = 1, num_points

      ! Check if point is within regcular grid
      ind_x = FLOOR((points(ind, 1) - x_axis(1)) * dx_inv) + 1
      IF ((ind_x < 1) .OR. (ind_x >= len_x)) THEN
        data_ip(ind) = value_ofb
        PRINT *, 'Warning: Out-of-bounds (x-axis):', ind, points(ind, 1)
        CONTINUE
      END IF
      ind_y = FLOOR((points(ind, 2) - y_axis(1)) * dy_inv) + 1
      IF ((ind_y < 1) .OR. (ind_y >= len_y)) THEN
        data_ip(ind) = value_ofb
        PRINT *, 'Warning: Out-of-bounds (y-axis):', ind, points(ind, 2)
        CONTINUE
      END IF

      ! Compute weights and interpolated value
      w_11 = (x_axis(ind_x + 1) - points(ind, 1)) * &
             (y_axis(ind_y + 1) - points(ind, 2))
      w_12 = (x_axis(ind_x + 1) - points(ind, 1)) * &
             (points(ind, 2) - y_axis(ind_y))
      w_21 = (points(ind, 1) - x_axis(ind_x)) * &
             (y_axis(ind_y + 1) - points(ind, 2))
      w_22 = (points(ind, 1) - x_axis(ind_x)) * &
             (points(ind, 2) - y_axis(ind_y))

      data_ip(ind) = (w_11 * data_in(ind_y, ind_x) + &
                      w_12 * data_in(ind_y + 1, ind_x) + &
                      w_21 * data_in(ind_y, ind_x + 1) + &
                      w_22 * data_in(ind_y + 1, ind_x + 1)) * &
                     (dx_inv * dy_inv)

    END DO
    !$OMP END PARALLEL DO
    time_2 = OMP_GET_WTIME()
    WRITE(6,*) 'Interpolation: ', time_2 - time_1, ' s'

  END SUBROUTINE bilinear

  ! ---------------------------------------------------------------------------
  ! Inverse distance weighted interpolation (unstructured -> regular grid)
  ! with k-d tree
  ! ---------------------------------------------------------------------------

  SUBROUTINE idw_kdtree(points, num_points, data_in, &
    x_axis, len_x, y_axis, len_y, num_neighbours, data_ip)

    INTEGER :: num_points, len_x, len_y
    REAL, DIMENSION(2, num_points) :: points ! (2, # of pts)
    REAL, DIMENSION(num_points) :: data_in
    REAL, DIMENSION(len_x) :: x_axis
    REAL, DIMENSION(len_y) :: y_axis
    INTEGER :: num_neighbours
    REAL, DIMENSION(len_y, len_x) :: data_ip

    !f2py threadsafe
    !f2py intent(in) points
    !f2py intent(hide) num_points
    !f2py intent(in) data_in
    !f2py intent(in) x_axis
    !f2py intent(hide) len_x
    !f2py intent(in) y_axis
    !f2py intent(hide) len_y
    !f2py intent(in) num_neighbours
    !f2py intent(out) data_ip

    INTEGER, DIMENSION(:), allocatable :: index
    TYPE(kdtree_type) :: tree
    REAL(8) :: time_1, time_2
    INTEGER :: i
    INTEGER :: ind_x, ind_y
    REAL, DIMENSION(2) :: point_target
    REAL, DIMENSION(:, :), allocatable :: neighbours
    INTEGER, DIMENSION(:), allocatable :: neighbours_index
    REAL :: numerator, denominator, dist

    ALLOCATE(index(num_points))
    index(:) = (/(i, i=1,num_points, 1)/)

    ! Build the k-d tree
    time_1 = OMP_GET_WTIME()
    CALL build_kd_tree(tree, points, size(points, 2), 0, index)
    time_2 = OMP_GET_WTIME()
    WRITE(6,*) 'Build tree: ', time_2 - time_1, ' s'

    ! Perform interpolation
    time_1 = OMP_GET_WTIME()
    ALLOCATE(neighbours(3, num_neighbours))
    ALLOCATE(neighbours_index(num_neighbours))
    !$OMP PARALLEL DO PRIVATE(ind_x, point_target, neighbours, &
    !$OMP neighbours_index, numerator, denominator, i, dist)
    DO ind_y = 1, len_y
      DO ind_x = 1, len_x

        ! Find nearest n neighbours
        point_target = [x_axis(ind_x), y_axis(ind_y)]
        neighbours = 1.0e6 ! Initialise neighbours array to large values
        CALL nearest_neighbours(tree%root, point_target, 0, neighbours, &
          num_neighbours, neighbours_index)

        ! Inverse distance weighted interpolation
        IF (sqrt(neighbours(3, 1)) <= (TINY(1.0) * 1e4)) THEN
          ! nearest neighbour extremely close -> avoid division by zero
          ! or (1.0 / dist) = Infinity
          data_ip(ind_y, ind_x) = data_in(neighbours_index(1))
          WRITE(6,*) 'Extremely close nearest neighbour (ind_x =', ind_x , &
            'ind_y = ', ind_y, ', dist = ', sqrt(neighbours(3, 1)), ')'
        ELSE
          numerator = 0.0
          denominator = 0.0
          DO i = 1, num_neighbours
            dist = sqrt(neighbours(3, i))
            numerator = numerator + (data_in(neighbours_index(i)) / dist)
            denominator = denominator + (1.0 / dist)
          END DO
          data_ip(ind_y, ind_x) = numerator / denominator
        END IF

      END DO
    END DO
    !$OMP END PARALLEL DO
    time_2 = OMP_GET_WTIME()
    WRITE(6,*) 'Interpolation: ', time_2 - time_1, ' s'

    ! Free the memory associated with the k-d tree
    CALL free_kdtree(tree)

    DEALLOCATE(index)
    DEALLOCATE(neighbours)
    DEALLOCATE(neighbours_index)

  END SUBROUTINE idw_kdtree

  ! ---------------------------------------------------------------------------
  ! Inverse distance weighted interpolation (unstructured -> regular grid)
  ! with nearest neighbours search for equally spaced regular grid
  ! ---------------------------------------------------------------------------

  SUBROUTINE idw_esrg_nearest(points, num_points, data_in, &
    x_axis, len_x, y_axis, len_y, grid_spac, num_neighbours, data_ip)

    INTEGER :: num_points, len_x, len_y
    REAL, DIMENSION(num_points, 2) :: points ! (# of pts, 2)
    REAL, DIMENSION(num_points) :: data_in
    REAL, DIMENSION(len_x) :: x_axis
    REAL, DIMENSION(len_y) :: y_axis
    REAL :: grid_spac
    INTEGER :: num_neighbours
    REAL, DIMENSION(len_y, len_x) :: data_ip

    !f2py threadsafe
    !f2py intent(in) points
    !f2py intent(hide) num_points
    !f2py intent(in) data_in
    !f2py intent(in) x_axis
    !f2py intent(hide) len_x
    !f2py intent(in) y_axis
    !f2py intent(hide) len_y
    !f2py intent(in) grid_spac
    !f2py intent(in) num_neighbours
    !f2py intent(out) data_ip

    INTEGER :: ind_x, ind_y
    REAL(8) :: time_1, time_2
    integer :: i
    REAL :: numerator, denominator
    INTEGER, DIMENSION(:), allocatable :: index_of_pts
    INTEGER, DIMENSION(:, :), allocatable :: indptr
    INTEGER, DIMENSION(:, :), allocatable :: num_ppgc
    INTEGER, DIMENSION(:), allocatable :: index_nn
    REAL, DIMENSION(:), allocatable :: dist_nn

    ALLOCATE(index_of_pts(num_points))
    ALLOCATE(indptr(len_y, len_x))
    ALLOCATE(num_ppgc(len_y, len_x))
    ALLOCATE(index_nn(num_neighbours))
    ALLOCATE(dist_nn(num_neighbours))

    time_1 = OMP_GET_WTIME()
    CALL assign_points_to_cells(points, x_axis, y_axis, grid_spac, &
      index_of_pts, indptr, num_ppgc)
    time_2 = OMP_GET_WTIME()
    WRITE(6,*) 'Assign points to cells: ', time_2 - time_1, ' s'

    ! Perform interpolation
    time_1 = OMP_GET_WTIME()
    !$OMP PARALLEL DO PRIVATE(ind_x, index_nn, dist_nn, numerator, denominator)
    DO ind_y = 1, len_y
      DO ind_x = 1, len_x

        ! Find nearest n neighbours
        CALL nearest_neighbours_esrg(points, &
          index_of_pts, indptr, num_ppgc, num_neighbours, &
          x_axis, y_axis, grid_spac, ind_x, ind_y, &
          index_nn, dist_nn)

        ! Inverse distance weighted interpolation
        IF (dist_nn(1) <= (TINY(1.0) * 1e4)) THEN
          ! nearest neighbour extremely close -> avoid division by zero
          ! or (1.0 / dist) = Infinity
          data_ip(ind_y, ind_x) = data_in(index_nn(1))
          WRITE(6,*) 'Extremely close nearest neighbour (ind_x =', ind_x , &
            'ind_y = ', ind_y, ', dist = ', dist_nn(1), ')'
        ELSE
          numerator = 0.0
          denominator = 0.0
          DO i = 1, num_neighbours
            numerator = numerator + (data_in(index_nn(i)) / dist_nn(i))
            denominator = denominator + (1.0 / dist_nn(i))
          END DO
          data_ip(ind_y, ind_x) = numerator / denominator
        END IF

      END DO
    END DO
    !$OMP END PARALLEL DO
    time_2 = OMP_GET_WTIME()
    WRITE(6,*) 'Interpolation: ', time_2 - time_1, ' s'

    DEALLOCATE(index_of_pts)
    DEALLOCATE(indptr)
    DEALLOCATE(num_ppgc)
    DEALLOCATE(index_nn)
    DEALLOCATE(dist_nn)

  END SUBROUTINE idw_esrg_nearest

  ! ---------------------------------------------------------------------------
  ! Inverse distance weighted interpolation (unstructured -> regular grid)
  ! with nearest neighbour search and points connected via triangulation
  ! for equally spaced regular grid
  ! ---------------------------------------------------------------------------

  SUBROUTINE idw_esrg_connected(points, num_points, data_in, &
    x_axis, len_x, y_axis, len_y, grid_spac, &
    indices_con, indices_con_len, indptr_con, data_ip)

    INTEGER :: num_points, len_x, len_y, indices_con_len
    REAL, DIMENSION(num_points, 2) :: points ! (# of pts, 2)
    REAL, DIMENSION(num_points) :: data_in
    REAL, DIMENSION(len_x) :: x_axis
    REAL, DIMENSION(len_y) :: y_axis
    REAL :: grid_spac
    INTEGER, DIMENSION(indices_con_len) :: indices_con
    INTEGER, DIMENSION(num_points + 1) :: indptr_con
    REAL, DIMENSION(len_y, len_x) :: data_ip

    !f2py threadsafe
    !f2py intent(in) points
    !f2py intent(hide) num_points
    !f2py intent(in) data_in
    !f2py intent(in) x_axis
    !f2py intent(hide) len_x
    !f2py intent(in) y_axis
    !f2py intent(hide) len_y
    !f2py intent(in) grid_spac
    !f2py intent(in) indices_con
    !f2py intent(hide) indices_con_len
    !f2py intent(in) indptr_con
    !f2py intent(out) data_ip

    INTEGER :: ind_x, ind_y
    REAL(8) :: time_1, time_2
    integer :: i
    REAL :: numerator, denominator, dist
    INTEGER, DIMENSION(:), allocatable :: index_of_pts
    INTEGER, DIMENSION(:, :), allocatable :: indptr
    INTEGER, DIMENSION(:, :), allocatable :: num_ppgc
    INTEGER, DIMENSION(:), allocatable :: index_nn
    REAL, DIMENSION(:), allocatable :: dist_nn
    REAL, DIMENSION(:) :: centre(2)

    ALLOCATE(index_of_pts(num_points))
    ALLOCATE(indptr(len_y, len_x))
    ALLOCATE(num_ppgc(len_y, len_x))
    ALLOCATE(index_nn(1))
    ALLOCATE(dist_nn(1))

    time_1 = OMP_GET_WTIME()
    CALL assign_points_to_cells(points, x_axis, y_axis, grid_spac, &
      index_of_pts, indptr, num_ppgc)
    time_2 = OMP_GET_WTIME()
    WRITE(6,*) 'Assign points to cells: ', time_2 - time_1, ' s'

    ! Perform interpolation
    time_1 = OMP_GET_WTIME()
    !$OMP PARALLEL DO PRIVATE(ind_x, centre, index_nn, dist_nn, numerator, &
    !$OMP denominator, i, dist)
    DO ind_y = 1, len_y
      DO ind_x = 1, len_x

      	! Centre coordinates of the grid cell
        centre(1) = x_axis(ind_x)
        centre(2) = y_axis(ind_y)

        ! Find nearest n neighbours
        CALL nearest_neighbours_esrg(points, &
          index_of_pts, indptr, num_ppgc, 1, &
          x_axis, y_axis, grid_spac, ind_x, ind_y, &
          index_nn, dist_nn)

        ! Inverse distance weighted interpolation
        IF (dist_nn(1) <= (TINY(1.0) * 1e4)) THEN
          ! nearest neighbour extremely close -> avoid division by zero
          ! or (1.0 / dist) = Infinity
          data_ip(ind_y, ind_x) = data_in(index_nn(1))
          WRITE(6,*) 'Extremely close nearest neighbour (ind_x =', ind_x , &
            'ind_y = ', ind_y, ', dist = ', dist_nn(1), ')'
        ELSE
          numerator = 0.0
          denominator = 0.0
          ! Nearest neighbour:
          numerator = numerator + (data_in(index_nn(1)) / dist_nn(1))
          denominator = denominator + (1.0 / dist_nn(1))
          ! Points connected via triangulation:
          DO i = indptr_con(index_nn(1)), (indptr_con(index_nn(1) + 1) - 1)
            dist = SQRT((centre(1) - points(indices_con(i), 1)) ** 2 &
                      + (centre(2) - points(indices_con(i), 2)) ** 2)
            numerator = numerator + (data_in(indices_con(i)) / dist)
            denominator = denominator + (1.0 / dist)
          END DO
          data_ip(ind_y, ind_x) = numerator / denominator
        END IF

      END DO
    END DO
    !$OMP END PARALLEL DO
    time_2 = OMP_GET_WTIME()
    WRITE(6,*) 'Interpolation: ', time_2 - time_1, ' s'

    DEALLOCATE(index_of_pts)
    DEALLOCATE(indptr)
    DEALLOCATE(num_ppgc)
    DEALLOCATE(index_nn)
    DEALLOCATE(dist_nn)

  END SUBROUTINE idw_esrg_connected

  ! ---------------------------------------------------------------------------

END MODULE interpolation
