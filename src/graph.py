from collections import defaultdict
from enum import Enum
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
plt.set_loglevel("info") 
import numpy as np


# Clear the log file at the start of the run
with open("graph.log", "w"):
    pass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s - %(message)s",
    handlers=[logging.FileHandler("graph.log"), logging.StreamHandler()],
)


@dataclass
class Node:
    def __init__(self, key, heuristic: float = 0.0, value=None, pos: tuple = None):
        self.key = key
        self.heuristic = heuristic  # Default heuristic key
        self.edges = []
        self.value = value


@dataclass
class Edge:
    def __init__(self, from_node: Node, to_node: Node, weight: float = 1):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight


class Stack:
    """A simple stack implementation."""

    def __init__(self):
        self.items = []

    def enqueue(self, item):  # Normally this would be called push
        # logging.debug(f"Pushing {item} onto stack")
        self.items.append(item)

    def dequeue(self):  # Normally this would be called pop
        if not self.is_empty():
            item = self.items.pop()
            # logging.debug(f"Popping {item} from stack")
            return item
        raise IndexError("pop from empty stack")

    def is_empty(self):
        return len(self.items) == 0

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")


class Queue:
    """A simple queue implementation."""

    def __init__(self):
        self.items = []

    def enqueue(self, item):
        # logging.debug(f"Enqueuing {item} to queue")
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            item = self.items.pop(0)
            # logging.debug(f"Dequeuing {item} from queue")
            return item
        raise IndexError("dequeue from empty queue")

    def is_empty(self):
        return len(self.items) == 0

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("peek from empty queue")


class PriorityQueue:
    """A simple priority queue implementation."""

    def __init__(self):
        self.items = []

    def enqueue(self, item, priority):
        # logging.debug(f"Enqueuing {item} with priority {priority}")
        self.items.append((item, priority))
        self.items.sort(
            key=lambda x: x[1]
        )  # Sort by priority, NOTE: this is O(n log n) for each enqueue, and can be improved assuming items are sorted beforehand

    def dequeue(self):
        if not self.is_empty():
            item, priority = self.items.pop(0)
            # logging.debug(f"Dequeuing {item} from queue")
            return item
        raise IndexError("dequeue from empty queue")

    def is_empty(self):
        return len(self.items) == 0

    def peek(self):
        if not self.is_empty():
            return self.items[0][0]
        raise IndexError("peek from empty queue")


class Graph:
    """A simple undirected graph implementation."""

    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)

    def add_node(self, key: str | int, value: any = None, pos: tuple = None):
        node = Node(key, value=value)
        self.nodes[key] = node
        # logging.debug(f"Node creation for v: {key}, node: {node}")

        return node

    def add_edge(self, from_key: str | int, to_key: str | int, weight=1):
        from_node, to_node = self.nodes.get(from_key), self.nodes.get(to_key)
        # logging.debug(
        #     f"Edge creation from {from_node.key} to {to_node.key} with weight {weight}"
        # )
        edge = Edge(from_node, to_node, weight)
        self.edges[from_node.key].append(edge)
        self.edges[to_node.key].append(
            edge
        )  # Assuming undirected graph, add edge in both directions
        return edge

    def load(self, filepath):
        logging.info(f"Loading graph from {filepath}")

        with open(filepath, "r") as f:
            for line in f.readlines():
                parts = line.strip().split(",")

                if parts[0] == "E":  # Edge
                    _, from_key, to_key, weight = parts

                    # Formatting
                    weight = float(weight.strip())
                    from_key = from_key.strip()
                    to_key = to_key.strip()

                    # Check if nodes exist, if not, create them
                    from_node = self.nodes.get(from_key, None) or self.add_node(
                        from_key
                    )
                    to_node = self.nodes.get(to_key, None) or self.add_node(to_key)

                    # Add edge
                    self.add_edge(from_node.key, to_node.key, weight)

                elif parts[0] == "H":  # Heuristic
                    _, node_key, heuristic = parts
                    node_key = node_key.strip()
                    heuristic = float(heuristic.strip())
                    if node_key in self.nodes:  # NODE DOES NOT ADD IF NOT EXISTS
                        self.nodes[node_key].heuristic = heuristic
                        # logging.debug(
                        #     f"Setting heuristic for node {node_key} to {heuristic}"
                        # )

    def vizualize(self, start=None, goal=None, path=None):
        import networkx as nx

        # import netwulf as netwulf

        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(node.key)
            # Add heuristic
            # G.nodes[node.key]["heuristic"] = node.heuristic
            # logging.debug(f"Node {node.key} has heuristic {node.heuristic}")
            # Show heuristic in node label
            # G.nodes[node.key]["label"] = f"{node.key} h={node.heuristic}"

        for edges in self.edges.values():
            for edge in edges:
                if not G.has_edge(edge.from_node.key, edge.to_node.key):
                    print(edge.from_node.key, edge.to_node.key, edge.weight)
                    G.add_edge(edge.from_node.key, edge.to_node.key, weight=edge.weight)
                    logging.debug(
                        f"Adding edge from {edge.from_node.key} to {edge.to_node.key} with weight {edge.weight}"
                    )
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, "weight")
        # Use the custom label for each node
        # node_labels = {n: G.nodes[n]["label"] for n in G.nodes}
        nx.draw(
            G,
            pos,
            # labels=node_labels,
            with_labels=True,
            node_color="lightblue",
            node_size=700,
            font_weight="bold",
        )
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Graph Visualization (Heuristic in Node Label)")
        plt.show()


class Maze2D(Graph):
    """
    Graph Representation of a 2D Maze.
    """

    def __init__(self):
        super().__init__()
        self.N = None
        self.M = None

    # @override
    def load(self, filepath):
        logging.info(f"Loading maze from {filepath}")

        with open(filepath, "r") as f:
            content = f.readlines()

            for y, line in enumerate(content):
                for x, char in enumerate(line.strip()):
                    x = int(x)
                    y = int(y)
                    key = (x, y)
                    if char == " ":
                        self.add_node(key)

                    elif char == "#":
                        self.add_node(key, value="wall")

                    left_node = self.nodes.get((x - 1, y), None)
                    top_node = self.nodes.get((x, y - 1), None)

                    if left_node and left_node.value != "wall" and char != "#":
                        self.add_edge(left_node.key, key)
                    if top_node and top_node.value != "wall" and char != "#":
                        self.add_edge(top_node.key, key)

            self.N = x
            self.M = y
            logging.info(f"Maze dimensions: {self.N}x{self.M}")

            print("All nodes created, now creating edges...")

    def vizualize(self, start=None, goal=None, path=None):
        import numpy as np

        for key, node in self.nodes.items():
            if start == key:
                plt.scatter(key[0], -key[1], color="green", s=100, label="Start")

            elif goal == key:
                plt.scatter(key[0], -key[1], color="purple", s=100, label="Goal")

            elif node.value == "wall":
                plt.scatter(key[0], -key[1], color="red", s=100)
            else:
                plt.scatter(key[0], -key[1], color="black", s=100)

        for edges in self.edges.values():
            for edge in edges:
                from_key = edge.from_node.key
                to_key = edge.to_node.key

                if edge.from_node.value != "wall" and edge.to_node.value != "wall":
                    color = "blue" if edge.weight == 1 else "yellow"
                    plt.plot(
                        [from_key[0], to_key[0]],
                        [-from_key[1], -to_key[1]],
                        color=color,
                        linestyle="dashed",
                    )

        # Draw path if provided
        if path:
            for i in range(len(path) - 1):
                from_key = path[i]
                to_key = path[i + 1]

                plt.plot(
                    [from_key[0], to_key[0]],
                    [-from_key[1], -to_key[1]],
                    color="orange",
                    linewidth=2,
                )
        plt.show()


class Maze2DInputMode(Enum):
    """
    Enum for input modes of the maze.
    """

    WALL = 1  # Wall mode
    SIZE = 2  # Size mode
    START = 3  # Start position mode
    GOAL = 4  # Goal position mode


class Maze3D(Graph):
    """
    Graph Representation of a 3D Maze.
    """

    def __init__(self):
        super().__init__()
        self.N = None
        self.M = None
        self.H = None
        self.start: Node = None
        self.goal: Node = None
        self.wall_width = 0.15 # Meters
        self.wall_length = 1.2 # Meters
        self.wall_height = 0.5 # Meters
        self.maze_width = 2.4 # Meters
        self.maze_length = 2.4 # Meters

        self.offset_x = -1.258
        self.offset_y = -4.749
        self.offset_z = 0.7

    # @override
    def load(self, filepath):
        logging.info(f"Loading 3D maze from {filepath}")

        x_min = 1
        y_min = 1
        z_min = 1
        x_max = 0
        y_max = 0
        z_max = 0

        input_mode = Maze2DInputMode.WALL
        with open(filepath, "r") as f:
            content = f.readlines()

            for line in content:
                if "size" in line:
                    input_mode = Maze2DInputMode.SIZE
                    continue

                elif line[0] == "%" or not line.strip():
                    continue

                elif input_mode == Maze2DInputMode.SIZE:
                    parts = line.strip().split(" ")
                    self.N, self.M, self.H = map(int, parts)
                    logging.info(f"3D Maze dimensions: {self.N}x{self.M}x{self.H}")
                    input_mode = Maze2DInputMode.START
                    continue

                elif input_mode == Maze2DInputMode.START:
                    parts = line.strip().split(" ")
                    start_x, start_y, start_z = map(int, parts)
                    key = (start_x, start_y, start_z)
                    self.start = self.add_node(key, value="start")
                    input_mode = Maze2DInputMode.GOAL

                elif input_mode == Maze2DInputMode.GOAL:
                    parts = line.strip().split(" ")
                    goal_x, goal_y, goal_z = map(int, parts)
                    key = (goal_x, goal_y, goal_z)
                    self.goal = self.add_node(key, value="goal")
                    input_mode = Maze2DInputMode.WALL
                    continue

                elif input_mode == Maze2DInputMode.WALL:

                    x, y, z = map(int, line.strip().split(" "))
                    key = (x, y, z)
                    print(f"Adding node {key}, value=wall")
                    self.add_node(key, value="wall")

        x_max, y_max, z_max = self.N, self.M, self.H
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for z in range(z_min, z_max + 1):
                    key = (x, y, z)
                    if key not in self.nodes:
                        self.add_node(key)

                    # Check neighbors in 3D space
                    # Only allow movement to 6-connected neighbors (no diagonals)
                    neighbors = [
                        (x + 1, y, z),
                        (x - 1, y, z),
                        (x, y + 1, z),
                        (x, y - 1, z),
                        (x, y, z + 1),
                        (x, y, z - 1),
                    ]
                    for neighbor in neighbors:
                        if neighbor in self.nodes and neighbor != key:
                            if (
                                self.nodes[key].value != "wall"
                                and self.nodes[neighbor].value != "wall"
                            ):
                                self.add_edge(key, neighbor)

    def vizualize(self, start=None, goal=None, path=None):
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np

        start = self.start.key if start is None else start
        goal = self.goal.key if goal is None else goal

        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        for key, node in self.nodes.items():
            if start == key:
                # Draw a 3D ball (sphere) for the start node
                u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                r = 0.4
                x = key[0] + r * np.cos(u) * np.sin(v)
                y = key[1] + r * np.sin(u) * np.sin(v)
                z = key[2] + r * np.cos(v)
                ax.plot_surface(x, y, z, color="green", alpha=0.5, linewidth=0)

            elif goal == key:
                # Draw a 3D ball (sphere) for the start node
                u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                r = 0.4
                x = key[0] + r * np.cos(u) * np.sin(v)
                y = key[1] + r * np.sin(u) * np.sin(v)
                z = key[2] + r * np.cos(v)
                ax.plot_surface(x, y, z, color="purple", alpha=0.5, linewidth=0)

            elif node.value == "wall":
                # Draw a cube for the wall at (x, y, z)

                x, y, z = key
                # Half the size for centering
                d = 1 / 2
                # 8 vertices of the cube
                vertices = [
                    [x - d, y - d, z - d],
                    [x + d, y - d, z - d],
                    [x + d, y + d, z - d],
                    [x - d, y + d, z - d],
                    [x - d, y - d, z + d],
                    [x + d, y - d, z + d],
                    [x + d, y + d, z + d],
                    [x - d, y + d, z + d]
                ]
                # Define the 6 faces using the vertices
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                    [vertices[4], vertices[7], vertices[3], vertices[0]]   # left
                ]
                cube = Poly3DCollection(
                    faces, facecolors="red", edgecolors="k", linewidths=0.5, alpha=1
                )
                ax.add_collection3d(cube)

        # for edges in self.edges.values():
        #     for edge in edges:
        #         from_key = edge.from_node.key
        #         to_key = edge.to_node.key

        #         # if edge.from_node.value != "wall" and edge.to_node.value != "wall":
        #         color = "blue" if edge.weight == 1 else "yellow"
        #         ax.plot(
        #             [from_key[0], to_key[0]],
        #             [from_key[1], to_key[1]],
        #             [from_key[2], to_key[2]],
        #             color=color,
        #             linestyle="dashed",
        #         )

        # Draw path if provided
        if path:
            for i in range(len(path) - 1):
                from_key = path[i]
                to_key = path[i + 1]
                ax.plot(
                    [from_key[0], to_key[0]],
                    [from_key[1], to_key[1]],
                    [from_key[2], to_key[2]],
                    color="blue",
                    linewidth=3 if path else 2,
                )

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_xlim(0, self.N+1)
        ax.set_ylim(0, self.M+1)
        ax.set_zlim(0, self.H+1)
        ax.axis("equal")

        ax.set_title("3D Maze Visualization")
        plt.show()

    def translate_coords(self, coord : tuple[float, float, float], offset_x : float = -1.258, offset_y : float = -4.749, offset_z : float = 0.7) -> tuple[float, float, float]:
        assert isinstance(offset_x, (int, float)), "offset_x must be a number"
        assert isinstance(offset_y, (int, float)), "offset_y must be a number"
        assert isinstance(offset_z, (int, float)), "offset_z must be a number"

        x, y, z = coord #unpack coordinates
        x, y, z = x-1, y-1, z-1 #adjust to 0-indexed coordinates
        n_walls_y = (self.M // 2) # Number of walls in y direction
        x_len = self.maze_length / self.N # Length of each cell in x direction

        #Y is a bit tricky, since it alternates between walls of 15 cm and cells of 1.2 M / 2 - wall_width
        
        width_walls = (self.wall_width * n_walls_y)

        # Maze width - width of walls divided by number of cells aka. non-wall cells
        y_len = (self.maze_width - width_walls) / (self.M - n_walls_y)

        # Z is simply the height of the wall
        z_len = self.wall_height

        #Center offsets are half the width, length and height of a cell
        center_offset_x = x_len / 2
        center_offset_y = y_len / 2
        center_offset_z = z_len / 2

        # Apply the offsets to the coordinates
        x = (x * x_len) + offset_x + center_offset_x
        y = (np.ceil(y/2) * y_len) + (y//2 * self.wall_width) + offset_y + center_offset_y
        z = (z * z_len) + offset_z + center_offset_z

        return (x, y, z)


    def translate_path(self, path, offset_x : float = 0, offset_y : float = 0, offset_z : float = 0):
        t_path = [self.translate_coords(coord, offset_x, offset_y, offset_z) for coord in path]
        return t_path
        


class search:
    """
    A class for search algorithms.
    Keeps track of visited nodes and the frontier of the search.
    Returns the path from the start node to the goal node if found.
    """

    @staticmethod
    def depth_first_search(
        graph: Graph, start_key: str | int, goal_key: str | int
    ) -> tuple[bool, list, set, Stack]:
        """
        Performs a depth-first search on the graph from start_key to goal_key.

        Args:
            graph (Graph): The graph to search.
            start_key (str | int): The key of the starting node.
            goal_key (str | int): The key of the goal node.
        Returns:
            tuple: A tuple containing:
                - bool: True if the goal is found, False otherwise.
                - list: The path from start to goal if found, empty list otherwise.
                - set: The set of visited nodes.
                - Stack: The stack used for the search.
        """

        stack = Stack()
        visited = set()
        parent_map = {}
        stack.enqueue(start_key)

        while not stack.is_empty():
            current_key = stack.dequeue()  # Get next node to explore from frontier
            if current_key == goal_key:  # Check if we reached the goal
                # Reconstruct path from start to goal
                path = []
                while current_key is not None:
                    path.append(current_key)
                    current_key = parent_map.get(current_key, None)
                return (
                    True,
                    path[::-1],
                    visited,
                    stack,
                )  # Return reversed path i.e. from start to goal

            if current_key not in visited:  # If we haven't visited this node yet
                visited.add(current_key)  # Mark as visited
                # Explore all edges from current node
                for edge in graph.edges[current_key]:
                    if edge.weight == float("inf"):
                        logging.debug(
                            f"Skipping edge from {edge.from_node.key} to {edge.to_node.key} with weight inf"
                        )
                        continue
                    # Logic a bit funny, but has to check both directions of the edge
                    next_node = (
                        edge.to_node.key
                        if edge.from_node.key == current_key
                        else edge.from_node.key
                    )  # Get the next node to explore
                    if next_node not in visited and next_node not in stack.items:
                        parent_map[next_node] = current_key
                        stack.enqueue(next_node)

                logging.debug(f"Current Node: {current_key}, stack: {stack.items}")
        return False, [], visited, stack  # Path not found

    @staticmethod
    def breadth_first_search(
        graph: Graph, start_key: str | int, goal_key: str | int
    ) -> tuple[bool, list, set, Queue]:
        """
        Performs a breadth-first search on the graph from start_key to goal_key.

        Args:
            graph (Graph): The graph to search.
            start_key (str | int): The key of the starting node.
            goal_key (str | int): The key of the goal node.
        Returns:
            tuple: A tuple containing:
                - bool: True if the goal is found, False otherwise.
                - list: The path from start to goal if found, empty list otherwise.
                - set: The set of visited nodes.
                - Queue: The queue used for the search.
        """

        queue = Queue()
        visited = set()
        parent_map = {}
        queue.enqueue(start_key)

        while not queue.is_empty():
            current_key = queue.dequeue()
            if current_key == goal_key:
                # Reconstruct path from start to goal
                path = []
                while current_key is not None:
                    path.append(current_key)
                    current_key = parent_map.get(current_key, None)
                return True, path[::-1], visited, queue
            if current_key not in visited:
                visited.add(current_key)
                for edge in graph.edges[current_key]:
                    if edge.weight == float("inf"):
                        logging.debug(
                            f"Skipping edge from {edge.from_node.key} to {edge.to_node.key} with weight inf"
                        )
                        continue
                    next_node = (
                        edge.to_node.key
                        if edge.from_node.key == current_key
                        else edge.from_node.key
                    )
                    if next_node not in visited and next_node not in queue.items:
                        parent_map[next_node] = current_key
                        queue.enqueue(next_node)
        return False, [], visited, queue

    @staticmethod
    def dikstra(
        graph: Graph, start_key: str | int, goal_key: str | int
    ) -> tuple[bool, list, set, PriorityQueue]:
        """
        Performs Dijkstra's algorithm on the graph from start_key to goal_key.

        Args:
            graph (Graph): The graph to search.
            start_key (str | int): The key of the starting node.
            goal_key (str | int): The key of the goal node.
        Returns:
            tuple: A tuple containing:
                - bool: True if the goal is found, False otherwise.
                - list: The path from start to goal if found, empty list otherwise.
                - set: The set of visited nodes.
                - PriorityQueue: The priority queue used for the search.
        """

        priority_queue = PriorityQueue()
        visited = set()
        parent_map = {}
        distances = {node.key: float("inf") for node in graph.nodes.values()}
        distances[start_key] = 0
        priority_queue.enqueue(start_key, 0)

        while not priority_queue.is_empty():
            current_key = priority_queue.dequeue()
            if current_key == goal_key:
                # Reconstruct path from start to goal
                path = []
                while current_key is not None:
                    path.append(current_key)
                    current_key = parent_map.get(current_key, None)
                return True, path[::-1], visited, priority_queue

            if current_key not in visited:
                visited.add(current_key)
                for edge in graph.edges[current_key]:
                    next_node = (
                        edge.to_node.key
                        if edge.from_node.key == current_key
                        else edge.from_node.key
                    )
                    new_distance = distances[current_key] + edge.weight
                    if new_distance < distances[next_node]:
                        distances[next_node] = new_distance
                        parent_map[next_node] = current_key
                        priority_queue.enqueue(next_node, new_distance)

        return False, [], visited, priority_queue

    @staticmethod
    def greedy_first_search(
        graph: Graph, start_key: str | int, goal_key: str | int
    ) -> tuple[bool, list, set, PriorityQueue]:
        """
        Performs a greedy first search on the graph from start_key to goal_key.

        Args:
            graph (Graph): The graph to search.
            start_key (str | int): The key of the starting node.
            goal_key (str | int): The key of the goal node.
        Returns:
            tuple: A tuple containing:
                - bool: True if the goal is found, False otherwise.
                - list: The path from start to goal if found, empty list otherwise.
                - set: The set of visited nodes.
                - PriorityQueue: The priority queue used for the search.
        """

        priority_queue = PriorityQueue()
        visited = set()
        parent_map = {}
        priority_queue.enqueue(start_key, graph.nodes[start_key].heuristic)

        while not priority_queue.is_empty():
            current_key = priority_queue.dequeue()
            if current_key == goal_key:
                # Reconstruct path from start to goal
                path = []
                while current_key is not None:
                    path.append(current_key)
                    current_key = parent_map.get(current_key, None)
                return True, path[::-1], visited, priority_queue

            if current_key not in visited:
                visited.add(current_key)
                for edge in graph.edges[current_key]:
                    next_node = (
                        edge.to_node.key
                        if edge.from_node.key == current_key
                        else edge.from_node.key
                    )
                    if (
                        next_node not in visited
                        and next_node not in priority_queue.items
                    ):
                        parent_map[next_node] = current_key
                        priority_queue.enqueue(
                            next_node, graph.nodes[next_node].heuristic
                        )

        return False, [], visited, priority_queue

    @staticmethod
    def a_star(
        graph: Graph, start_key: str | int, goal_key: str | int
    ) -> tuple[bool, list, set, PriorityQueue]:
        """
        Performs A* search on the graph from start_key to goal_key.

        Args:
            graph (Graph): The graph to search.
            start_key (str | int): The key of the starting node.
            goal_key (str | int): The key of the goal node.
        Returns:
            tuple: A tuple containing:
                - bool: True if the goal is found, False otherwise.
                - list: The path from start to goal if found, empty list otherwise.
                - set: The set of visited nodes.
                - PriorityQueue: The priority queue used for the search.
        """

        priority_queue = PriorityQueue()
        visited = set()
        parent_map = {}
        g_scores = {node.key: float("inf") for node in graph.nodes.values()}
        g_scores[start_key] = 0
        f_scores = {node.key: float("inf") for node in graph.nodes.values()}
        f_scores[start_key] = graph.nodes[start_key].heuristic
        priority_queue.enqueue(start_key, f_scores[start_key])

        while not priority_queue.is_empty():
            current_key = priority_queue.dequeue()
            if current_key == goal_key:
                # Reconstruct path from start to goal
                path = []
                while current_key is not None:
                    path.append(current_key)
                    current_key = parent_map.get(current_key, None)
                return True, path[::-1], visited, priority_queue

            if current_key not in visited:
                visited.add(current_key)
                for edge in graph.edges[current_key]:
                    next_node = (
                        edge.to_node.key
                        if edge.from_node.key == current_key
                        else edge.from_node.key
                    )
                    tentative_g_score = g_scores[current_key] + edge.weight

                    if tentative_g_score < g_scores[next_node]:
                        parent_map[next_node] = current_key
                        g_scores[next_node] = tentative_g_score
                        f_scores[next_node] = (
                            tentative_g_score + graph.nodes[next_node].heuristic
                        )
                        priority_queue.enqueue(next_node, f_scores[next_node])

        return False, [], visited, priority_queue

    @staticmethod
    def evaluate_search_algorithm(
        algorithm,
        graph: Graph,
        start_key: str | int,
        goal_key: str | int,
        viz: bool = False,
    ):
        """
        Evaluates a search algorithm on the graph. And logs the formattet results.

        Args:
            algorithm (function): The search algorithm to evaluate.
            graph (Graph): The graph to search.
            start_key (str | int): The key of the starting node.
            goal_key (str | int): The key of the goal node.
        Returns:
            None
        """

        found, path, visited, frontier = algorithm(graph, start_key, goal_key)
        print()

        logging.info(f"Search Algorithm: {algorithm.__name__}")
        logging.info(f"Start Node: {start_key}, Goal Node: {goal_key}")
        logging.info(f"Goal Found: {found}")
        logging.info(
            f"Path: {' -> '.join(map(str, path)) if path else 'No path found'}"
        )
        logging.info(f"Visited Nodes: {len(visited)}")
        logging.info(f"Frontier Size: {len(frontier.items)}")

        if viz:
            logging.info("Visualizing the graph...")
            graph.vizualize(start_key, goal_key, path)
            logging.info("Graph visualization complete.")


if __name__ == "__main__":
    # FILEPATH = 'src/examples/figure2_graph.csv'
    # graph = Graph()
    # graph.load(FILEPATH)
    # logging.info(f'Graph loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges.')

    # search.evaluate_search_algorithm(search.depth_first_search, graph, 's0', 's23')

    # search.evaluate_search_algorithm(search.breadth_first_search, graph, 's0', 's23')

    # graph.vizualize()
    # logging.info('Graph visualization complete.')

    # FILEPATH = 'src/examples/figure4_graph.csv'
    # graph = Graph()
    # graph.load(FILEPATH)
    # logging.info(f'Graph loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges.')

    # graph.vizualize() # Does not work after Python 3.12, distutils are deprecated
    # logging.info('Graph visualization complete.')

    # search.evaluate_search_algorithm(search.depth_first_search, graph, 's0', 's23')

    # search.evaluate_search_algorithm(search.breadth_first_search, graph, 's0', 's23')

    # search.evaluate_search_algorithm(search.dikstra, graph, 's0', 's23')

    # search.evaluate_search_algorithm(search.greedy_first_search, graph, 's0', 's23')

    # search.evaluate_search_algorithm(search.a_star, graph, 's0', 's23')

    # FILEPATH = "src/examples/maze2d_v2.txt"
    # maze = Maze2D()
    # maze.load(FILEPATH)
    # logging.info(
    #     f"Maze loaded with {len(maze.nodes)} nodes and {len(maze.edges)} edges."
    # )

    # start, goal = (1, 10), (1, 6)

    # search.evaluate_search_algorithm(search.depth_first_search, maze, start, goal, viz=True)

    # search.evaluate_search_algorithm(search.breadth_first_search, maze, start, goal, viz=True)

    # search.evaluate_search_algorithm(search.dikstra, maze, start, goal, viz=True)

    # search.evaluate_search_algorithm(search.greedy_first_search, maze, start, goal, viz=True)

    # search.evaluate_search_algorithm(search.a_star, maze, start, goal, viz=True)

    FILEPATH = "data/wall.txt"

    maze = Maze3D()
    maze.load(FILEPATH)
    logging.info(
        f"3D Maze loaded with {len(maze.nodes)} nodes and {len(maze.edges)} edges."
    )

    search.evaluate_search_algorithm(
        search.depth_first_search, maze, maze.start.key, maze.goal.key, viz=True
    )

    maze.translate_coords((1,1,1))

    print("Done")

    # maze.vizualize()
