! Description: Functions to interpolate between regular and unstructured grid
!              (e.g. triangle mesh)
!
! Author: Christian Steger, October 2024

MODULE interpolation

  USE kd_tree
  IMPLICIT NONE

  CONTAINS

  ! ---------------------------------------------------------------------------
  ! Bilinear interpolation (regular -> unstructured grid)
  ! ---------------------------------------------------------------------------

  SUBROUTINE bilinear(x_axis, dim_x, y_axis, dim_y, data_in, &
    points, num_points, data_ip)

    INTEGER :: dim_x, dim_y, num_points
    REAL, DIMENSION(dim_x) :: x_axis
    REAL, DIMENSION(dim_y) :: y_axis
    REAL, DIMENSION(dim_y, dim_x) :: data_in
    REAL, DIMENSION(num_points, 2) :: points
    REAL, DIMENSION(num_points) :: data_ip

    !f2py threadsafe
    !f2py intent(in) x_axis
    !f2py intent(hide) dim_x
    !f2py intent(in) y_axis
    !f2py intent(hide) dim_y
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

    ! Compute inverse of x- and y-grid spacing
    delta = 0.0
    DO ind = 1, (dim_x - 1)
      delta = delta + (x_axis(ind + 1) - x_axis(ind))
    END DO
    dx_inv = 1.0 / (delta / REAL(dim_x - 1))
    delta = 0.0
    DO ind = 1, (dim_y - 1)
      delta = delta + (y_axis(ind + 1) - y_axis(ind))
    END DO
    dy_inv = 1.0 / (delta / REAL(dim_y - 1))

    ! Perform interpolation
    DO ind = 1, num_points

      ! Check if point is within regcular grid
      ind_x = FLOOR((points(ind, 1) - x_axis(1)) * dx_inv) + 1
      IF ((ind_x < 1) .OR. (ind_x >= dim_x)) THEN
        data_ip(ind) = value_ofb
        PRINT *, 'Warning: Out-of-bounds (x-axis):', ind, points(ind, 1)
        CONTINUE
      END IF
      ind_y = FLOOR((points(ind, 2) - y_axis(1)) * dy_inv) + 1
      IF ((ind_y < 1) .OR. (ind_y >= dim_y)) THEN
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

  END SUBROUTINE bilinear

  ! ---------------------------------------------------------------------------
  ! Inverse distance weighted interpolation (unstructured -> regular grid)
  ! with k-d tree
  ! ---------------------------------------------------------------------------

  SUBROUTINE idw_kdtree(points, num_points, data_in, &
    x_axis, dim_x, y_axis, dim_y, num_neighbours, data_ip)

    INTEGER :: num_points, dim_x, dim_y
    REAL, DIMENSION(2, num_points) :: points
    REAL, DIMENSION(num_points) :: data_in
    REAL, DIMENSION(dim_x) :: x_axis
    REAL, DIMENSION(dim_y) :: y_axis
    INTEGER :: num_neighbours
    REAL, DIMENSION(dim_y, dim_x) :: data_ip

    !f2py threadsafe
    !f2py intent(in) points
    !f2py intent(hide) num_points
    !f2py intent(in) data_in
    !f2py intent(in) x_axis
    !f2py intent(hide) dim_x
    !f2py intent(in) y_axis
    !f2py intent(hide) dim_y
    !f2py intent(in) num_neighbours
    !f2py intent(out) data_ip

    INTEGER, DIMENSION(:), allocatable :: index
    TYPE(kdtree_type) :: tree
    REAL :: time_1, time_2
    integer :: i
    integer :: ind_x, ind_y
    REAL, DIMENSION(2) :: point_target
    REAL, DIMENSION(:, :), allocatable :: neighbours
    INTEGER, DIMENSION(:), allocatable :: neighbours_index
    REAL :: numerator, denominator, dist

    ALLOCATE(index(num_points))
    index(:) = (/(i, i=1,num_points, 1)/)

    ! Build the k-d tree
    CALL cpu_time(time_1)
    CALL build_kd_tree(tree, points, size(points, 2), 0, index)
    CALL cpu_time(time_2)
    WRITE(6,*) 'Build tree: ', time_2 - time_1

    ! Perform interpolation
    ALLOCATE(neighbours(3, num_neighbours))
    ALLOCATE(neighbours_index(num_neighbours))
    CALL cpu_time(time_1)
    DO ind_y = 1, dim_y
      DO ind_x = 1, dim_x

				! Find nearest n neighbours
        point_target = [x_axis(ind_x), y_axis(ind_y)]
        neighbours = 1.0e6 ! Initialise neighbours array to large values
        CALL nearest_neighbours(tree%root, point_target, 0, neighbours, &
          num_neighbours, neighbours_index)

				! Nearest neighbour interpolation
			  !data_ip(ind_y, ind_x) = data_in(neighbours_index(1))

			  ! Inverse distance weighted interpolation
        numerator = 0.0
        denominator = 0.0
        DO i = 1, num_neighbours
          dist = sqrt(neighbours(3, i))
          numerator = numerator + (data_in(neighbours_index(i)) / dist)
          denominator = denominator + (1.0 / dist)
        END DO
        data_ip(ind_y, ind_x) = numerator / denominator

      END DO
    END DO
    CALL cpu_time(time_2)
    WRITE(6,*) 'Interpolation: ', time_2 - time_1

    ! Free the memory associated with the k-d tree
    CALL free_kdtree(tree)

    DEALLOCATE(index)
    DEALLOCATE(neighbours)
    DEALLOCATE(neighbours_index)

  END SUBROUTINE idw_kdtree

  ! ---------------------------------------------------------------------------

END MODULE interpolation
