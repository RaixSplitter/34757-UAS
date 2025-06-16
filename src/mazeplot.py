'''import matplotlib.pyplot as plt

def is_on_wall(x, y, walls):
    """
    Check if a point (x, y) lies exactly on any wall line.
    """
    for (x1, y1), (x2, y2) in walls:
        # Check for horizontal wall
        if y1 == y2 and y == y1 and min(x1, x2) <= x <= max(x1, x2):
            return True
        # Check for vertical wall
        if x1 == x2 and x == x1 and min(y1, y2) <= y <= max(y1, y2):
            return True
    return False

def draw_maze_with_nodes(walls, size=(11, 11)):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.set_aspect('equal')
    ax.set_xticks(range(size[0] + 1))
    ax.set_yticks(range(size[1] + 1))
    ax.grid(True, which='both')

    # Draw outer boundary
    ax.plot([0, size[0]], [0, 0], 'k', linewidth=2)
    ax.plot([0, size[0]], [size[1], size[1]], 'k', linewidth=2)
    ax.plot([0, 0], [0, size[1]], 'k', linewidth=2)
    ax.plot([size[0], size[0]], [0, size[1]], 'k', linewidth=2)

    # Draw internal walls
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        ax.plot([x1, x2], [y1, y2], 'k', linewidth=2)

    # Plot nodes only if not on a wall
    for x in range(size[0]):
        for y in range(size[1]):
            if not is_on_wall(x, y, walls):
                ax.plot(x, y, 'ro', markersize=4)

    plt.tight_layout()
    plt.show()

# Example wall list
walls = [
    ((0, 0), (10, 0)),
    ((0, 0), (0, 10)),
#    ((0, 11), (11, 11)),
#    ((11, 0), (11, 11)),
    ((0, 2), (7, 2)),
    ((7, 2), (7, 3)),
    ((0, 3), (3, 3)),
    ((3, 3), (3, 2))
]

draw_maze_with_nodes(walls, size=(11, 11))

'''

import matplotlib.pyplot as plt
from nodes_to_csv import export_edges_to_csv
def is_on_wall(x, y, walls):
    for (x1, y1), (x2, y2) in walls:
        if y1 == y2 and y == y1 and min(x1, x2) <= x <= max(x1, x2):
            return True
        if x1 == x2 and x == x1 and min(y1, y2) <= y <= max(y1, y2):
            return True
    return False

def has_wall_between(p1, p2, walls):
    x1, y1 = p1
    x2, y2 = p2

    # Horizontal neighbor
    if y1 == y2 and abs(x1 - x2) == 1:
        wall_x = min(x1, x2) + 1e-6  # handle edge cases
        return is_on_wall(wall_x, y1, walls)
    # Vertical neighbor
    if x1 == x2 and abs(y1 - y2) == 1:
        wall_y = min(y1, y2) + 1e-6
        return is_on_wall(x1, wall_y, walls)

    return True  # not a valid neighbor (not adjacent)

def build_nodes_and_edges(walls, size=(11, 11)):
    nodes = {}
    edges = {}
    node_counter = 0

    # Build nodes
    for y in range(size[1]):
        for x in range(size[0]):
            if not is_on_wall(x, y, walls):
                label = f's{node_counter}'
                nodes[label] = (x, y)
                edges[label] = []
                node_counter += 1

    # Build edges
    for label, (x, y) in nodes.items():
        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]
        for nx, ny in neighbors:
            for target_label, pos in nodes.items():
                if pos == (nx, ny) and not has_wall_between((x, y), (nx, ny), walls):
                    edges[label].append(target_label)

    return nodes, edges

def draw_maze_with_edges(nodes, edges, walls, size=(11, 11)):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.set_aspect('equal')
    ax.set_xticks(range(size[0] + 1))
    ax.set_yticks(range(size[1] + 1))
    ax.grid(True, which='both')

    # Draw walls
    for (x1, y1), (x2, y2) in walls:
        ax.plot([x1, x2], [y1, y2], 'k', linewidth=2)

    # Draw nodes and labels
    for label, (x, y) in nodes.items():
        ax.plot(x, y, 'ro', markersize=4)
        ax.text(x + 0.2, y + 0.2, label, fontsize=8, color='blue')

    # Draw valid edges
    for label, neighbors in edges.items():
        x1, y1 = nodes[label]
        for nlabel in neighbors:
            x2, y2 = nodes[nlabel]
            ax.plot([x1, x2], [y1, y2], 'g--', linewidth=1)

    plt.tight_layout()
    plt.show()

def get_node_coordinates_dict(walls, size=(11, 11)):
    """Returns a dict of node labels to (x, y, z=1) coordinates."""
    nodes, _ = build_nodes_and_edges(walls, size)
    node_coords = {label: (x, y, 1) for label, (x, y) in nodes.items()}
    return node_coords

def export_node_coordinates(walls, size=(11, 11), output_path='node_coordinates.py'):
    """
    Builds the node coordinates dictionary and writes it to a Python file
    as a variable named NODE_COORDINATES. Also returns the dictionary.
    """
    nodes, _ = build_nodes_and_edges(walls, size)
    node_coords = {label: (x-1, y-1, 1) for label, (x, y) in nodes.items()}

    with open(output_path, 'w') as f:
        f.write("NODE_COORDINATES = {\n")
        for label, (x, y, z) in node_coords.items():
            f.write(f"    '{label}': ({x}, {y}, {z}),\n")
        f.write("}\n")

    print(f"[✓] Exported NODE_COORDINATES to {output_path}")
    return node_coords


    

# Example wall layout
walls = [
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

# Build graph
nodes, edges = build_nodes_and_edges(walls, size=(11, 11))

# Visualize
draw_maze_with_edges(nodes, edges, walls, size=(11, 11))

print("Possible moves from s0:", edges.get('s0', []))

export_edges_to_csv(nodes, edges, filename='maze_graph.csv')




