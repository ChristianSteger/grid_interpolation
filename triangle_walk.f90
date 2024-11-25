! Description: Algorithm to find nearest neighbours in two-dimensional
!              equally spaced (dx = dy) regular grid (esrg)
!
! Author: Christian R. Steger, November 2024

MODULE triangle_walk

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: point, find_triangle, fill_nearest_neighbour

  TYPE :: point
    REAL :: x
    REAL :: y
  END TYPE point

  CONTAINS

  ! ---------------------------------------------------------------------------
  ! Find triangle containing point (or closest point on convex hull in case
  ! point is located outside of the mesh)
  ! ---------------------------------------------------------------------------

  SUBROUTINE find_triangle(points, simplices, neighbours, neighbour_none, &
    point_target, ind_tri, point_inside_ch, iters)

    REAL, DIMENSION(:, :), INTENT(IN) :: points ! (# of pts, 2)
    INTEGER, DIMENSION(:, :), INTENT(IN) :: simplices ! (# of triangles, 3)
    INTEGER, DIMENSION(:, :), INTENT(IN) :: neighbours ! (# of triangles, 3)
    INTEGER, INTENT(IN) :: neighbour_none
    TYPE(point), INTENT(IN) :: point_target
    INTEGER, INTENT(INOUT) :: ind_tri
    LOGICAL, INTENT(OUT) :: point_inside_ch
    INTEGER, INTENT(OUT) :: iters

    INTEGER :: ind_1, ind_2, ind_tri_pre
    LOGICAL :: point_outside
    INTEGER :: i
    TYPE(point), DIMENSION(2) :: line_edge

    iters = 0
    point_inside_ch = .TRUE.

    DO

      iters = iters + 1

      ! Find intersection with triangle edges
      point_outside = .FALSE.
      DO i = 1, 3
        ind_1 = i
        ind_2 = MODULO(i, 3) + 1
        line_edge(1)%x = points(simplices(ind_tri, ind_1), 1)
        line_edge(1)%y = points(simplices(ind_tri, ind_1), 2)
        line_edge(2)%x = points(simplices(ind_tri, ind_2), 1)
        line_edge(2)%y = points(simplices(ind_tri, ind_2), 2)
        IF (triangle_point_outside(line_edge, point_target)) THEN
          point_outside = .TRUE.
          EXIT
        END IF
      END DO
      IF (point_outside .EQV. .FALSE.) THEN
        !WRITE(6,*) 'Exit: Triangle containing point found'
        EXIT
      END IF
      i = i - 1
      IF (i < 1) THEN
        i = 3
      END IF
      ind_tri_pre = ind_tri
      ind_tri = neighbours(ind_tri, i)

      ! Points outside of convex hull
      IF (ind_tri == neighbour_none) THEN
        !WRITE(6,*) 'Point is outside of convex hull'
        ind_tri = ind_tri_pre  ! set triangle index to last valid
        point_inside_ch = .FALSE.
        EXIT

      END IF

    END DO

  END SUBROUTINE find_triangle

  ! ---------------------------------------------------------------------------
  ! Fill cells in rectangular grid with nearest neighbour
  ! ---------------------------------------------------------------------------

  SUBROUTINE fill_nearest_neighbour(mask_outside, x_axis, y_axis, data_ip)

    LOGICAL, DIMENSION(:, :), INTENT(IN) :: mask_outside
    REAL, DIMENSION(:), INTENT(IN) :: x_axis
    REAL, DIMENSION(:), INTENT(IN) :: y_axis
    REAL, DIMENSION(:, :), INTENT(INOUT) :: data_ip

    INTEGER :: ind_y, ind_x, i, j
    INTEGER :: level
    LOGICAL :: nn_found
    REAL :: dist_sq_sel = HUGE(1.0)
    REAL :: radius_sq_min, radius_sq_max, dist_sq

    DO ind_y = 1, SIZE(y_axis)
      DO ind_x = 1, SIZE(x_axis)
        IF (mask_outside(ind_y, ind_x)) THEN
          level = 0
          nn_found = .FALSE.
          dist_sq_sel = HUGE(1.0)
          DO WHILE (.NOT. nn_found)
            level = level + 1
            radius_sq_min = (REAL(level) - 0.5) ** 2
            radius_sq_max = (REAL(level) + 0.5) ** 2
            DO i = MAX(ind_y - level, 1), MIN(ind_y + level, SIZE(y_axis))
              DO j = MAX(ind_x - level, 1), MIN(ind_x + level, SIZE(x_axis))
                IF (.NOT. mask_outside(i, j)) THEN
                  dist_sq = REAL((i - ind_y) ** 2 + (j - ind_x) ** 2)
                  IF ((dist_sq >= radius_sq_min) &
                    .AND. (dist_sq < radius_sq_max) &
                    .AND. (dist_sq < dist_sq_sel)) THEN
                    dist_sq_sel = dist_sq
                    data_ip(ind_y, ind_x) = data_ip(i, j)
                    nn_found = .TRUE.
                 END IF
                END IF
              END DO
            END DO
          END DO
        END IF
      END DO
    END DO

  END SUBROUTINE fill_nearest_neighbour

  ! ---------------------------------------------------------------------------
  ! Auxiliary functions
  ! ---------------------------------------------------------------------------

  ! Checks, based on a single edge of a triangle, if a point is outside of
  ! the triangle (the vertices must be oriented counterclockwise)
  FUNCTION triangle_point_outside(edge, point_in) RESULT(res)
    TYPE(point), DIMENSION(2), INTENT(IN) :: edge
    TYPE(point), INTENT(IN) :: point_in
    LOGICAL :: res

    REAL margin, cross_prod

    ! margin = 0.0
    margin = -1e-10  ! 'safety margin' (0.0 or slightly smaller)
    cross_prod = (edge(2)%x - edge(1)%x) * (point_in%y - edge(1)%y) &
      - (edge(2)%y - edge(1)%y) * (point_in%x - edge(1)%x)
    res = (cross_prod < margin)

  END FUNCTION

END MODULE triangle_walk
