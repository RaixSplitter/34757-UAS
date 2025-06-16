# coordinates.py
from mazeplot import export_node_coordinates


walls = [  # Walls of the maze
    ((0, 0), (10, 0)),
    ((0, 0), (0, 10)),
    ((0, 2), (7, 2)),
    ((7, 2), (7, 3)),
    ((0, 3), (3, 3)),
    ((3, 3), (3, 2)),
    ((9, 2), (9, 5)),
    ((9, 5), (3, 5)),
    ((3, 5), (3, 6)),
    ((3, 6), (2, 6)),
    ((0, 4), (1, 4)),
    ((1, 4), (1, 3)),
    ((1, 3), (3, 3)),
    ((5, 5), (5, 4)),
    ((7, 5), (7, 7)),
    ((0, 8), (2, 8)),
    ((2, 8), (2, 9)),
    ((4, 9), (9, 9)),
    ((4, 9), (4, 8)),
    ((4, 8), (5, 8)),
    ((5, 7), (5, 9)),
    ((9, 9), (9, 7))
]
# A dictionary mapping node IDs to their (x, y, z) coordinates
NODE_COORDINATES = export_node_coordinates(walls, size=(11, 11), output_path='node_coordinates.py')


def path_to_matlab_matrix(path: list[str], node_to_coord: dict[str, tuple[int, int, int]]) -> str:
    """
    Converts a list of node IDs to a MATLAB-friendly matrix string.
    
    Args:
        path: List of node names like ['s0', 's1', ...]
        node_to_coord: Dictionary mapping node names to (x, y, z) coordinates

    Returns:
        A string like '[0 0 1; 4 0 1; ...]' suitable for MATLAB.
    """
    rows = [f"{x} {y} {z}" for node in path if (coord := node_to_coord.get(node)) for x, y, z in [coord]]
    matlab_str = "[" + "; ".join(rows) + "]"
    return matlab_str


def get_coordinate(node_id: str) -> tuple[int, int]:
    """Returns the (x, y) coordinate of a node ID."""
    return NODE_COORDINATES.get(node_id, None)

def path_to_coordinates(path: list[str]) -> list[tuple[int, int]]:
    """Converts a list of node IDs to a list of coordinates."""
    return [get_coordinate(node) for node in path]

print(path_to_matlab_matrix(['s0', 's1'], NODE_COORDINATES))