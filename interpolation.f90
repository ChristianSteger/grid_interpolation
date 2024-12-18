! Description: Functions to interpolate between regular and unstructured grid
!              (e.g. triangle mesh)
!
! Author: Christian R. Steger, November 2024

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
    REAL, DIMENSION(num_points, 2) :: points
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
    REAL :: weight_11, weight_12, weight_21, weight_22
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
    !$OMP PARALLEL DO PRIVATE(ind_x, ind_y, weight_11, weight_12, &
    !$OMP weight_21, weight_22)
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
      weight_11 = (x_axis(ind_x + 1) - points(ind, 1)) * &
        (y_axis(ind_y + 1) - points(ind, 2))
      weight_12 = (x_axis(ind_x + 1) - points(ind, 1)) * &
        (points(ind, 2) - y_axis(ind_y))
      weight_21 = (points(ind, 1) - x_axis(ind_x)) * &
        (y_axis(ind_y + 1) - points(ind, 2))
      weight_22 = (points(ind, 1) - x_axis(ind_x)) * &
        (points(ind, 2) - y_axis(ind_y))

      data_ip(ind) = (weight_11 * data_in(ind_y, ind_x) + &
        weight_12 * data_in(ind_y + 1, ind_x) + &
        weight_21 * data_in(ind_y, ind_x + 1) + &
        weight_22 * data_in(ind_y + 1, ind_x + 1)) * (dx_inv * dy_inv)

    END DO
    !$OMP END PARALLEL DO
    time_2 = OMP_GET_WTIME()
    WRITE(*,'(a,f5.3,a)') 'Interpolation: ' , time_2 - time_1, ' s'

  END SUBROUTINE bilinear

  ! ---------------------------------------------------------------------------
  ! Inverse distance weighted interpolation (unstructured -> regular grid)
  ! with k-d tree
  ! ---------------------------------------------------------------------------

  SUBROUTINE idw_kdtree(points, num_points, data_in, &
    x_axis, len_x, y_axis, len_y, num_neighbours, data_ip)

    INTEGER :: num_points, len_x, len_y
    REAL, DIMENSION(2, num_points) :: points
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
    WRITE(*,'(a,f5.3,a)') 'Tree building: ' , time_2 - time_1, ' s'

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
    WRITE(*,'(a,f5.3,a)') 'Interpolation: ' , time_2 - time_1, ' s'

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
    REAL, DIMENSION(num_points, 2) :: points
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
    WRITE(*,'(a,f5.3,a)') 'Assign points to cells: ' , time_2 - time_1, ' s'

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
    WRITE(*,'(a,f5.3,a)') 'Interpolation: ' , time_2 - time_1, ' s'

    DEALLOCATE(index_of_pts)
    DEALLOCATE(indptr)
    DEALLOCATE(num_ppgc)
    DEALLOCATE(index_nn)
    DEALLOCATE(dist_nn)

  END SUBROUTINE idw_esrg_nearest

  ! ---------------------------------------------------------------------------
  ! Barycentric interpolation (triangle mesh -> regular grid) with triangle
  ! walk. For points lying outside of the convex hull, interpolation is
  ! performed for the closest location on the triangle mesh. Termination of
  ! the 'triangle walk' algorithm is only guaranteed for meshes obtained
  ! with Delaunay triangulation
  ! ---------------------------------------------------------------------------

  SUBROUTINE barycentric_interpolation(points, num_points, data_in, &
    x_axis, len_x, y_axis, len_y, &
    simplices, neighbours, num_triangles, data_ip)

    INTEGER :: num_points, num_triangles, len_x, len_y
    REAL, DIMENSION(num_points, 2) :: points
    REAL, DIMENSION(num_points) :: data_in
    REAL, DIMENSION(len_x) :: x_axis
    REAL, DIMENSION(len_y) :: y_axis
    INTEGER, DIMENSION(num_triangles, 3) :: simplices
    INTEGER, DIMENSION(num_triangles, 3) :: neighbours
    REAL, DIMENSION(len_y, len_x) :: data_ip

    !f2py threadsafe
    !f2py intent(in) points
    !f2py intent(hide) num_points
    !f2py intent(in) data_in
    !f2py intent(in) x_axis
    !f2py intent(hide) len_x
    !f2py intent(in) y_axis
    !f2py intent(hide) len_y
    !f2py intent(in) simplices
    !f2py intent(in) neighbours
    !f2py intent(hide) num_triangles
    !f2py intent(out) data_ip

    INTEGER :: ind_x, ind_y, i
    TYPE(point) :: point_target
    INTEGER :: iters, ind_tri, ind_tri_start
    INTEGER :: neighbour_none
    TYPE(point) :: vertex_1, vertex_2, vertex_3
    REAL :: denom, weight_vt1, weight_vt2, weight_vt3
    INTEGER, DIMENSION(3) :: ind
    REAL(8) :: time_1, time_2
    INTEGER :: iters_tot
    LOGICAL :: point_inside_ch
    LOGICAL, DIMENSION(len_y, len_x) :: mask_outside
    REAL :: opt_x, opt_y, centroid_x, centroid_y
    REAL :: dist_sq_min, dist_sq

    neighbour_none = 0  ! value for no neighbour
    mask_outside(:, :) = .FALSE.

    ! Compute optimal starting position for triangle walk
    ind_tri_start = 1
    time_1 = OMP_GET_WTIME()
    opt_x = x_axis(1)
    opt_y = SUM(y_axis) / REAL(SIZE(y_axis))
    dist_sq_min = HUGE(1.0)
    DO i = 1, num_points
      centroid_x = SUM(points(simplices(i, :), 1)) / 3.0
      centroid_y = SUM(points(simplices(i, :), 2)) / 3.0
      dist_sq = (centroid_x - opt_x) ** 2 + (centroid_y - opt_y) ** 2
      IF (dist_sq < dist_sq_min) THEN
        dist_sq_min = dist_sq
        ind_tri_start = i
      END IF
    END DO
    time_2 = OMP_GET_WTIME()
    WRITE(*,'(a,f5.3,a)') 'Determine optimal start: ' , time_2 - time_1, ' s'

    ! Perform interpolation
    iters_tot = 0
    time_1 = OMP_GET_WTIME()
    !$OMP PARALLEL DO PRIVATE(ind_x, point_target, ind_tri, point_inside_ch, &
    !$OMP iters, ind, vertex_1, vertex_2, vertex_3, denom, &
    !$OMP weight_vt1, weight_vt2, weight_vt3) REDUCTION(+:iters_tot)
    DO ind_y = 1, len_y
      ind_tri = ind_tri_start
      DO ind_x = 1, len_x

        ! Find triangle
        point_target%x = x_axis(ind_x)
        point_target%y = y_axis(ind_y)
        CALL find_triangle(points, simplices, neighbours, neighbour_none, &
          point_target, ind_tri, point_inside_ch, iters)
        iters_tot = iters_tot + iters

        ! Barycentric interpolation
        IF (point_inside_ch) THEN
          ind = simplices(ind_tri, :)
          vertex_1%x = points(ind(1), 1)
          vertex_1%y = points(ind(1), 2)
          vertex_2%x = points(ind(2), 1)
          vertex_2%y = points(ind(2), 2)
          vertex_3%x = points(ind(3), 1)
          vertex_3%y = points(ind(3), 2)
          denom = (vertex_2%y - vertex_3%y) * (vertex_1%x - vertex_3%x) &
            + (vertex_3%x - vertex_2%x) * (vertex_1%y - vertex_3%y)
          weight_vt1 = ((vertex_2%y - vertex_3%y) &
            * (point_target%x - vertex_3%x) &
            + (vertex_3%x - vertex_2%x) * (point_target%y - vertex_3%y)) &
            / denom
          weight_vt2 = ((vertex_3%y - vertex_1%y) &
            * (point_target%x - vertex_3%x) &
            + (vertex_1%x - vertex_3%x) * (point_target%y - vertex_3%y)) &
            / denom
          weight_vt3 = 1.0 - weight_vt1 - weight_vt2
          data_ip(ind_y, ind_x) = data_in(ind(1)) * weight_vt1 &
            + data_in(ind(2)) * weight_vt2 &
            + data_in(ind(3)) * weight_vt3
        ELSE
          mask_outside(ind_y, ind_x) = .TRUE.
        END IF

      END DO
    END DO
    !$OMP END PARALLEL DO
    time_2 = OMP_GET_WTIME()
    WRITE(*,'(a,f5.3,a)') 'Interpolation: ' , time_2 - time_1, ' s'
    WRITE(*,'(a,i10)') 'Number of iterations:', iters_tot

    ! Fill missing cells in rectangular grid with nearest neighbour
    time_1 = OMP_GET_WTIME()
    CALL fill_nearest_neighbour(mask_outside, x_axis, y_axis, data_ip)
    time_2 = OMP_GET_WTIME()
    WRITE(*,'(a,f5.3,a)') 'Fill missing values: ' , time_2 - time_1, ' s'

  END SUBROUTINE barycentric_interpolation

  ! ---------------------------------------------------------------------------

END MODULE interpolation
