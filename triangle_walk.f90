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
    ind_tri_start, point_target, ind_tri_out, point_out, iters)

    REAL, DIMENSION(:, :), INTENT(IN) :: points ! (# of pts, 2)
    INTEGER, DIMENSION(:, :), INTENT(IN) :: simplices ! (# of triangles, 3)
    INTEGER, DIMENSION(:, :), INTENT(IN) :: neighbours ! (# of triangles, 3)
    INTEGER, INTENT(IN) :: neighbour_none
    INTEGER, INTENT(IN) :: ind_tri_start
    TYPE(point), INTENT(IN) :: point_target
    INTEGER, INTENT(OUT) :: ind_tri_out
    TYPE(point), INTENT(OUT) :: point_out
    INTEGER, INTENT(OUT) :: iters
    
    INTEGER, DIMENSION(4) :: ind_loop
    LOGICAL, DIMENSION(:), ALLOCATABLE :: tri_visited
    INTEGER :: ind_tri, ind_1, ind_2, ind_tri_pre, ind_rot, ind_opp, ind_rem
    LOGICAL :: inters_found
    INTEGER :: i, num_tri
    TYPE(point) :: centroid, point_base
    TYPE(point), DIMENSION(2) :: line_walk, line_edge
    REAL :: u, dist_sq_1, dist_sq_2
    
    ind_loop = (/ 1, 2, 3, 1 /)
    num_tri = SIZE(simplices,1) ! # of triangles  
    ALLOCATE(tri_visited(num_tri))
    tri_visited(:) = .FALSE.
    ind_tri = ind_tri_start
    inters_found = .TRUE.
    iters = 0

 		DO WHILE (.TRUE.)
    
      IF (.NOT. tri_visited(ind_tri)) THEN
        tri_visited(ind_tri) = .TRUE.
      ELSE
      	WRITE(6,*) 'Exit: Triangle already visited'
        ind_tri_out = ind_tri
        point_out = point_target
        EXIT
      END IF
      iters = iters + 1
      centroid%x = SUM(points(simplices(ind_tri, :), 1)) / 3.0
      centroid%y = SUM(points(simplices(ind_tri, :), 2)) / 3.0
      line_walk(1) = centroid
      line_walk(2) = point_target

      ! Find intersection with triangle edges
      inters_found = .FALSE.
      DO i = 1, 3
        ind_1 = ind_loop(i)
        ind_2 = ind_loop(i + 1)
        line_edge(1)%x = points(simplices(ind_tri, ind_1), 1)
        line_edge(1)%y = points(simplices(ind_tri, ind_1), 2)
        line_edge(2)%x = points(simplices(ind_tri, ind_2), 1)
        line_edge(2)%y = points(simplices(ind_tri, ind_2), 2)
        IF (linesegments_intersect(line_walk, line_edge)) THEN
          inters_found = .TRUE.
          EXIT
        END IF
      END DO
      IF (inters_found .EQV. .FALSE.) THEN
        WRITE(6,*) 'Exit: Triangle containing point found'
        ind_tri_out = ind_tri
        point_out = point_target
        EXIT
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
          ind_tri_out = ind_tri
          point_out = point_base
          EXIT
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
          DO WHILE (.TRUE.)

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
              ind_tri_out = ind_tri
              point_out = point_base
              EXIT
						END IF
						
            ! Check if point should be assigned to 'rotation vertices'
            dist_sq_1 = distance_sq(point_target, line_edge(1))
            dist_sq_2 = distance_sq(point_target, line_edge(2))
						IF (dist_sq_1 < dist_sq_2) THEN
              WRITE(6,*) 'Exit: Point is not perpendicular to edge -> use nearest vertices'
              ind_tri_out = ind_tri
              point_out%x = points(simplices(ind_tri, ind_rot), 1)
              point_out%y = points(simplices(ind_tri, ind_rot), 2)
              EXIT
						END IF
						
            ! Move to next line of convex hull during next iteration
            ind_opp = ind_rot
            ind_rot = ind_rem
          
          END DO
          
          EXIT ! point assigned in inner 'while loop' -> break out of outer

        END IF
        
      END IF

    END DO

    DEALLOCATE(tri_visited)

  END SUBROUTINE find_triangle

  ! ---------------------------------------------------------------------------
  ! Two-dimensional line/point algorithms and auxiliary functions
  ! ---------------------------------------------------------------------------

  ! Check if points a, b and c are counterclockwise oriented
  FUNCTION ccw(a, b, c) RESULT(res)
    TYPE(point), INTENT(IN) :: a, b, c
    LOGICAL :: res

    res = (c%y - a%y) * (b%x - a%x) > (b%y - a%y) * (c%x - a%x)

  END FUNCTION

  ! Check if line segments intersect (~returns .FALSE. if line segments only
  ! touch)
  FUNCTION linesegments_intersect(line_1, line_2) RESULT(res)
    TYPE(point), DIMENSION(2), INTENT(IN) :: line_1, line_2
    LOGICAL :: res

    res = ccw(line_1(1), line_2(1), line_2(2)) &
      .NEQV. ccw(line_1(2), line_2(1), line_2(2)) &
      .AND. ccw(line_1(1), line_1(2), line_2(1)) &
      .NEQV. ccw(line_1(1), line_1(2), line_2(2))

  END FUNCTION

  ! Find the base point, which marks the shortest distance bewteen 
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

  ! Get remaining element from array/list with 3 elements
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

    dist_sq = (A%x - B%x) ** 2 + (A%y - B%y) ** 2

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
    DO WHILE (.TRUE.)
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
