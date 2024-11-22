! Description: Algorithm to find nearest neighbours in two-dimensional
!              equally spaced (dx = dy) regular grid (esrg)
!
! Author: Christian R. Steger, November 2024

MODULE triangle_walk

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: find_triangle

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
    ind_tri, point_target, point_out, iters)

    REAL, DIMENSION(:, :), INTENT(IN) :: points ! (# of pts, 2)
    INTEGER, DIMENSION(:, :), INTENT(IN) :: simplices ! (# of triangles, 3)
    INTEGER, DIMENSION(:, :), INTENT(IN) :: neighbours ! (# of triangles, 3)
    INTEGER, INTENT(IN) :: neighbour_none
    INTEGER, INTENT(INOUT) :: ind_tri
    TYPE(point), INTENT(IN) :: point_target
    TYPE(point), INTENT(OUT) :: point_out
    INTEGER, INTENT(OUT) :: iters

    INTEGER :: ind_1, ind_2, ind_tri_pre, ind_rot, ind_opp, ind_rem
    LOGICAL :: point_outside
    INTEGER :: i
    TYPE(point) :: point_base
    TYPE(point), DIMENSION(2) :: line_edge
    REAL :: u, dist_sq_1, dist_sq_2

    iters = 0

    outer: DO

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
        WRITE(6,*) 'Exit: Triangle containing point found'
        point_out = point_target
        EXIT outer
      END IF
      i = i - 1
      IF (i < 1) THEN
        i = 3
      END IF
      ind_tri_pre = ind_tri
      ind_tri = neighbours(ind_tri, i)

      ! Handle points outside of convex hull
      IF (ind_tri == neighbour_none) THEN
        WRITE(6,*) 'Point is outside of convex hull'
        ind_tri = ind_tri_pre  ! set triangle index to last valid
        CALL base_point(line_edge, point_target, u, point_base)
        IF ((u >= 0) .AND. (u <= 1)) THEN
          ! -------------------------------------------------------------------
          ! Point is perpendicular to 'direct outer' edge of triangle
          ! -------------------------------------------------------------------
          WRITE(6,*) 'Exit: Point is perpendicular to direct outer edge of triangle'
          point_out = point_base
          EXIT outer
        ELSE
          ! -------------------------------------------------------------------
          ! Point is not perpendicular to 'direct outer' edge of triangle
          ! -------------------------------------------------------------------
          WRITE(6,*) 'Point is not perpendicular to direct outer edge of triangle'

          ! Define 'rotation' and 'opposite' vertices
          dist_sq_1 = distance_sq(point_target, line_edge(1))
          dist_sq_2 = distance_sq(point_target, line_edge(2))
          IF (dist_sq_1 < dist_sq_2) THEN
            ind_rot = ind_1
            ind_opp = ind_2
          ELSE
            ind_rot = ind_2
            ind_opp = ind_1
         END IF

          ! Move along line segments of convex hull
          DO

            ! Move to triangle that represents adjacent outer edge
            CALL get_adjacent_edge(simplices, neighbours, neighbour_none, &
              ind_tri, ind_rot, ind_opp, ind_rem)

            line_edge(1)%x = points(simplices(ind_tri, ind_rot), 1)
            line_edge(1)%y = points(simplices(ind_tri, ind_rot), 2)
            line_edge(2)%x = points(simplices(ind_tri, ind_rem), 1)
            line_edge(2)%y = points(simplices(ind_tri, ind_rem), 2)

            ! Check if point is perpendicular to edge
            CALL base_point(line_edge, point_target, u, point_base)
            IF ((u >= 0) .AND. (u <= 1)) THEN
              WRITE(6,*) 'Exit: Point is perpendicular to edge'
              point_out = point_base
              EXIT outer
            END IF

            ! Check if point should be assigned to 'rotation vertices'
            dist_sq_1 = distance_sq(point_target, line_edge(1))
            dist_sq_2 = distance_sq(point_target, line_edge(2))
            IF (dist_sq_1 < dist_sq_2) THEN
              WRITE(6,*) 'Exit: Point is not perpendicular to edge -> use nearest vertices'
              point_out%x = points(simplices(ind_tri, ind_rot), 1)
              point_out%y = points(simplices(ind_tri, ind_rot), 2)
              EXIT outer
            END IF

            ! Move to next line of convex hull during next iteration
            ind_opp = ind_rot
            ind_rot = ind_rem

          END DO

        END IF

      END IF

    END DO outer

  END SUBROUTINE find_triangle

  ! ---------------------------------------------------------------------------
  ! Two-dimensional line/point algorithms and auxiliary functions
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

  ! Find the base point, which marks the shortest distance between
  ! a line and a point)
  SUBROUTINE base_point(line, point_in, u, point_base)

    TYPE(point), DIMENSION(2), INTENT(IN) :: line
    TYPE(point), INTENT(IN) :: point_in
    REAL, INTENT(OUT) :: u
    TYPE(point), INTENT(OUT) :: point_base

    REAL :: dist_x, dist_y

    dist_x = line(2)%x - line(1)%x
    dist_y = line(2)%y - line(1)%y
    u = (dist_x * (point_in%x - line(1)%x) &
      + dist_y * (point_in%y - line(1)%y)) / (dist_x ** 2 + dist_y ** 2)
    point_base%x = line(1)%x + dist_x * u
    point_base%y = line(1)%y + dist_y * u

  END SUBROUTINE base_point

  ! Get remaining element from array with 3 elements
  FUNCTION get_rem(array, elem_1, elem_2) RESULT(elem_3)

    INTEGER, DIMENSION(3), INTENT(IN) :: array
    INTEGER, INTENT (IN) :: elem_1, elem_2
    INTEGER :: elem_3

    INTEGER :: i

    DO i = 1, 3
      IF ((array(i) /= elem_1) .AND. (array(i) /= elem_2)) THEN
        elem_3 = array(i)
        EXIT
      END IF
    END DO

  END FUNCTION

  ! Compute squared distance between point a and b
  FUNCTION distance_sq(a, b) RESULT(dist_sq)
    TYPE(point), INTENT(IN) :: a, b
    REAL :: dist_sq

    dist_sq = (a%x - b%x) ** 2 + (a%y - b%y) ** 2

  END FUNCTION

  ! Move to ajdacent edge in convex hull
  SUBROUTINE get_adjacent_edge(simplices, neighbours, neighbour_none, &
    ind_tri, ind_rot, ind_opp, ind_rem)

    INTEGER, DIMENSION(:, :), INTENT(IN) :: simplices ! (# of triangles, 3)
    INTEGER, DIMENSION(:, :), INTENT(IN) :: neighbours ! (# of triangles, 3)
    INTEGER, INTENT(IN) :: neighbour_none
    INTEGER, INTENT(INOUT) :: ind_tri, ind_rot, ind_opp
    INTEGER, INTENT(OUT) :: ind_rem

    INTEGER :: ind_vtx_rot, ind_vtx_opp, ind_vtx_rem

    ind_vtx_rot = simplices(ind_tri, ind_rot)  ! constant
    DO
      ind_vtx_opp = simplices(ind_tri, ind_opp)
      ind_vtx_rem = get_rem(simplices(ind_tri, :), ind_vtx_rot, ind_vtx_opp)
      IF (neighbours(ind_tri, ind_opp) == neighbour_none) THEN
        EXIT
      ELSE
        ind_tri = neighbours(ind_tri, ind_opp)
        ind_opp = findloc(simplices(ind_tri, :), value=ind_vtx_rem, dim=1)
      END IF
    END DO
    ind_rot = findloc(simplices(ind_tri, :), value=ind_vtx_rot, dim=1)
    ind_rem = findloc(simplices(ind_tri, :), value=ind_vtx_rem, dim=1)

  END SUBROUTINE get_adjacent_edge

END MODULE triangle_walk
