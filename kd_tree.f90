! Description: K-d tree implementation to find nearest neighbours in
!              two-dimensional space
!
! Author: Christian R. Steger, November 2024

MODULE kd_tree

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: kd_node, kdtree_type, build_kd_tree, nearest_neighbours, &
    free_kdtree

  TYPE :: kd_node
    REAL :: point(2)
    INTEGER :: index
    TYPE(kd_node), POINTER :: left => NULL()
    TYPE(kd_node), POINTER :: right => NULL()
  END TYPE kd_node

  TYPE :: kdtree_type
    TYPE(kd_node), POINTER :: root => NULL()
  END TYPE kdtree_type

  CONTAINS

  ! ---------------------------------------------------------------------------
  ! Build k-d tree from set of points
  ! ---------------------------------------------------------------------------

  RECURSIVE SUBROUTINE build_kd_tree(tree, points, num_points, depth, index)

    TYPE(kdtree_type), INTENT(INOUT) :: tree
    REAL, DIMENSION(:, :), INTENT(IN) :: points ! (2, # of pts)
    INTEGER, INTENT(IN) :: num_points
    INTEGER, INTENT(IN) :: depth
    INTEGER, DIMENSION(:), INTENT(IN) :: index

    INTEGER :: axis, ind_median
    REAL, DIMENSION(:, :), ALLOCATABLE :: sorted_points
    TYPE(kd_node), POINTER :: node
    INTEGER, DIMENSION(:), ALLOCATABLE :: sorted_index

    IF (num_points <= 0) RETURN

    axis = mod(depth, 2)

    ! Allocate and initialise arrays
    ALLOCATE(sorted_points(2, num_points))
    sorted_points = points
    ALLOCATE(sorted_index(num_points))
    sorted_index = index

    ! Sort data according to median
    ind_median = (num_points + 1) / 2
    CALL partition_iter(sorted_points, 1, num_points, ind_median, axis + 1, &
      sorted_index)

    ! Set node
    ALLOCATE(node)
    node%point = sorted_points(:, ind_median)
    node%index = sorted_index(ind_median)

    ! Build the left and right subtrees
    CALL build_kd_tree_sub(node%left, sorted_points(:, 1:ind_median - 1), &
      ind_median - 1, depth + 1, sorted_index(1:ind_median - 1))
    CALL build_kd_tree_sub(node%right, sorted_points(:, &
      ind_median + 1:num_points), num_points - ind_median, depth + 1, &
      sorted_index(ind_median + 1:num_points))

    ! Assign the node as the root if it is the first call
    IF (.not. associated(tree%root)) THEN
      tree%root => node
    ENDIF

  END SUBROUTINE build_kd_tree

  RECURSIVE SUBROUTINE build_kd_tree_sub(node, points, num_points, depth, &
    index)

    REAL, DIMENSION(:, :), INTENT(IN) :: points ! (2, # of pts)
    INTEGER, INTENT(IN) :: num_points
    INTEGER, INTENT(IN) :: depth
    INTEGER, DIMENSION(:), INTENT(IN) :: index

    TYPE(kd_node), POINTER :: node
    TYPE(kdtree_type) :: tree

    IF (num_points > 0) THEN
      CALL build_kd_tree(tree, points, num_points, depth, index)
      node => tree%root
    ELSE
      node => NULL()
    ENDIF

  END SUBROUTINE build_kd_tree_sub

  RECURSIVE SUBROUTINE partition_iter(points, ind_left, ind_right, ind_k, &
    axis, index)

    REAL, DIMENSION(:, :), INTENT(INOUT) :: points ! (2, # of pts)
    INTEGER, INTENT(IN) :: ind_left, ind_right, ind_k, axis
    INTEGER, DIMENSION(:), INTENT(INOUT) :: index

    INTEGER :: ind_pivot, ind_store
    REAL :: value_pivot
    REAL :: temp_1d
    REAL :: temp_2d(2)
    INTEGER :: ind
    INTEGER :: i

    ! No processing required for one point
    IF (ind_left == ind_right) RETURN

    ! Choose pivot element
    ind_pivot = (ind_left + ind_right) / 2
    value_pivot = points(axis, ind_pivot)

    ! Move pivot to the end
    temp_2d = points(:, ind_pivot)
    points(:, ind_pivot) = points(:, ind_right)
    points(:, ind_right) = temp_2d
    temp_1d = index(ind_pivot)
    index(ind_pivot) = index(ind_right)
    index(ind_right) = temp_1d

    ! Partitioning
    ind_store = ind_left
    DO ind = ind_left, (ind_right - 1)
      IF (points(axis, ind) < value_pivot) THEN
        temp_2d = points(:, ind_store)
        points(:, ind_store) = points(:, ind)
        points(:, ind) = temp_2d
        temp_1d = index(ind_store)
        index(ind_store) = index(ind)
        index(ind) = temp_1d
        ind_store = ind_store + 1
      ENDIF
    END DO

    ! Move pivot to final place
    temp_2d = points(:, ind_store)
    points(:, ind_store) = points(:, ind_right)
    points(:, ind_right) = temp_2d
    temp_1d = index(ind_store)
    index(ind_store) = index(ind_right)
    index(ind_right) = temp_1d

    ! Partition sub-array
    IF (ind_k < ind_store) THEN
      CALL partition_iter(points, ind_left, ind_store - 1, ind_k, axis, index)
    ELSE IF (ind_k > ind_store) THEN
      CALL partition_iter(points, ind_store + 1, ind_right, ind_k, axis, index)
    ENDIF

  END SUBROUTINE partition_iter

  ! ---------------------------------------------------------------------------
  ! Query tree for nearest neighbour(s)
  ! ---------------------------------------------------------------------------

  RECURSIVE SUBROUTINE nearest_neighbours(node, point_target, depth, &
    neighbours, num_neighbours, neighbours_index)

    REAL, DIMENSION(2), INTENT(IN) :: point_target
    INTEGER, INTENT(IN) :: depth
    REAL, DIMENSION(:, :), INTENT(INOUT) :: neighbours
    INTEGER, INTENT(IN) :: num_neighbours
    INTEGER, DIMENSION(:), INTENT(INOUT) :: neighbours_index

    TYPE(kd_node), POINTER :: node
    REAL :: dist_sqrt, diff
    INTEGER :: axis

    IF (.not. associated(node)) RETURN

    ! Calculate squared distance to the current node
    dist_sqrt = distance_squared(node%point, point_target)

    ! Insert the current point if it is one of the closest
    CALL insert_neighbour(neighbours, dist_sqrt, node%point, node%index, &
      num_neighbours, neighbours_index)

    axis = mod(depth, 2)
    diff = point_target(axis+1) - node%point(axis+1)

    ! Traverse the k-d tree based on the target point's value relative to the
    ! splitting axis
    IF (diff < 0.0) THEN
      CALL nearest_neighbours(node%left, point_target, depth + 1, &
        neighbours, num_neighbours, neighbours_index)
      IF (abs(diff) < maxval(neighbours(3, :))) THEN
        CALL nearest_neighbours(node%right, point_target, depth + 1, &
          neighbours, num_neighbours, neighbours_index)
      ENDIF
    ELSE
      CALL nearest_neighbours(node%right, point_target, depth + 1, &
        neighbours, num_neighbours, neighbours_index)
      IF (abs(diff) < maxval(neighbours(3, :))) THEN
        CALL nearest_neighbours(node%left, point_target, depth + 1, &
          neighbours, num_neighbours, neighbours_index)
      ENDIF
    ENDIF

  END SUBROUTINE nearest_neighbours

  ! Function to insert a neighbour into the neighbours array
  ! (array is kept sorted by distance)
  SUBROUTINE insert_neighbour(neighbours, dist_sqrt, point, index, &
    num_neighbours, neighbours_index)

    REAL, DIMENSION(:, :), INTENT(INOUT) :: neighbours
    REAL, INTENT(IN) :: dist_sqrt
    REAL, INTENT(IN) :: point(2)
    INTEGER, INTENT(IN) :: index
    INTEGER, INTENT(IN) :: num_neighbours
    INTEGER, DIMENSION(:), INTENT(INOUT) :: neighbours_index

    INTEGER :: i

    ! Find position to insert if the distance is smaller than the maximum
    ! stored distance
    IF ((size(neighbours, 2) < num_neighbours) .or. &
      (dist_sqrt < maxval(neighbours(3, :)))) THEN
      DO i = min(num_neighbours, size(neighbours, 2)), 2, -1
        IF (dist_sqrt < neighbours(3, i - 1)) THEN
          neighbours(:, i) = neighbours(:, i - 1)
          neighbours_index(i) = neighbours_index(i - 1)
        ELSE
          EXIT
        ENDIF
      END DO

      ! Insert the new point and its distance
      neighbours(1:2, i) = point
      neighbours(3, i) = dist_sqrt
      neighbours_index(i) = index

    ENDIF

  END SUBROUTINE insert_neighbour

  ! Squared distance between points
  PURE FUNCTION distance_squared(point1, point2) result(dist_sqrt)

    REAL, INTENT(IN) :: point1(2), point2(2)

    REAL :: dist_sqrt

    dist_sqrt = (point1(1) - point2(1)) ** 2 + (point1(2) - point2(2)) ** 2

  END FUNCTION distance_squared

  ! ---------------------------------------------------------------------------
  ! Deallocate memory associated with k-d tree
  ! ---------------------------------------------------------------------------

  RECURSIVE SUBROUTINE deallocate_kd_tree(node)

    TYPE(kd_node), POINTER :: node

    ! If the node is not associated, there is nothing to deallocate
    IF (.not. associated(node)) RETURN

    ! Recursively deallocate the left and right subtrees
    CALL deallocate_kd_tree(node%left)
    CALL deallocate_kd_tree(node%right)

    ! Nullify pointers to the children
    node%left => NULL()
    node%right => NULL()

    DEALLOCATE(node)

  END SUBROUTINE deallocate_kd_tree

  SUBROUTINE free_kdtree(tree)

    TYPE(kdtree_type), INTENT(INOUT) :: tree

    ! Deallocate the entire tree starting from the root
    CALL deallocate_kd_tree(tree%root)

    ! Nullify the root pointer of the tree
    tree%root => NULL()

  END SUBROUTINE free_kdtree

  ! ---------------------------------------------------------------------------

END MODULE kd_tree
