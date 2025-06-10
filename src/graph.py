from collections import defaultdict
from enum import Enum
import logging
from dataclasses import dataclass


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph.log'),
        logging.StreamHandler()
    ]
)


@dataclass
class Node:
    def __init__(self, value, heuristic: float = 0.0):
        self.value = value
        self.heuristic = heuristic # Default heuristic value
        self.edges = []
        
        
@dataclass
class Edge:
    def __init__(self, from_node : Node, to_node : Node, weight : float = 1):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight


class Stack:
    """A simple stack implementation."""
    def __init__(self):
        self.items = []


    def enqueue(self, item): #Normally this would be called push
        logging.debug(f"Pushing {item} onto stack")
        self.items.append(item)

    def dequeue(self): #Normally this would be called pop
        if not self.is_empty():
            item = self.items.pop()
            logging.debug(f"Popping {item} from stack")
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
        logging.debug(f"Enqueuing {item} to queue")
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            item = self.items.pop(0)
            logging.debug(f"Dequeuing {item} from queue")
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
        logging.debug(f"Enqueuing {item} with priority {priority}")
        self.items.append((item, priority))
        self.items.sort(key=lambda x: x[1])  # Sort by priority, NOTE: this is O(n log n) for each enqueue, and can be improved assuming items are sorted beforehand

    def dequeue(self):
        if not self.is_empty():
            item, priority = self.items.pop(0)
            logging.debug(f"Dequeuing {item} from queue")
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

    def add_node(self, value : str | int):
        logging.debug(f"Node creation for v: {value}")
        node = Node(value)
        self.nodes[value] = node
        return node

    def add_edge(self, from_value: str | int, to_value: str | int, weight=1):
        from_node, to_node = self.nodes.get(from_value.value), self.nodes.get(to_value.value)
        logging.debug(f"Edge creation from {from_node.value} to {to_node.value} with weight {weight}")
        edge = Edge(from_node, to_node, weight)
        self.edges[from_node.value].append(edge)
        self.edges[to_node.value].append(edge)  # Assuming undirected graph, add edge in both directions
        return edge
    
    def load(self, filepath):
        logging.info(f'Loading graph from {filepath}')
        
        with open(filepath, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                
                if parts[0] == 'E': #Edge
                    _, from_value, to_value, weight = parts
                    
                    #Formatting
                    weight = float(weight.strip())
                    from_value = from_value.strip()
                    to_value = to_value.strip()

                    # Check if nodes exist, if not, create them
                    from_node = self.nodes.get(from_value, None) or self.add_node(from_value)
                    to_node = self.nodes.get(to_value, None) or self.add_node(to_value)

                    # Add edge
                    self.add_edge(from_node, to_node, weight)
                
                elif parts[0] == 'H': # Heuristic
                    _, node_value, heuristic = parts
                    node_value = node_value.strip()
                    heuristic = float(heuristic.strip())
                    if node_value in self.nodes: # NODE DOES NOT ADD IF NOT EXISTS
                        self.nodes[node_value].heuristic = heuristic
                        logging.debug(f"Setting heuristic for node {node_value} to {heuristic}")
                    

    def vizualize(self):
        import networkx as nx
        # import netwulf as netwulf
        import matplotlib.pyplot as plt

        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(node.value)
            # Add heuristic
            G.nodes[node.value]['heuristic'] = node.heuristic
            logging.debug(f"Node {node.value} has heuristic {node.heuristic}")
            # Show heuristic in node label
            G.nodes[node.value]['label'] = f"{node.value} h={node.heuristic}"

        for edges in self.edges.values():
            for edge in edges:
                if not G.has_edge(edge.from_node.value, edge.to_node.value):
                    G.add_edge(edge.from_node.value, edge.to_node.value, weight=edge.weight)
                    logging.debug(f"Adding edge from {edge.from_node.value} to {edge.to_node.value} with weight {edge.weight}")

        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        # Use the custom label for each node
        node_labels = {n: G.nodes[n]['label'] for n in G.nodes}
        nx.draw(G, pos, labels=node_labels, with_labels=True, node_color='lightblue', node_size=700, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Graph Visualization (Heuristic in Node Label)")
        plt.show()


class search:
    """
    A class for search algorithms.
    Keeps track of visited nodes and the frontier of the search.
    Returns the path from the start node to the goal node if found.
    """
    
    @staticmethod
    def depth_first_search(graph: Graph, start_value: str | int, goal_value: str | int) -> tuple[bool, list, set, Stack]:
        """
        Performs a depth-first search on the graph from start_value to goal_value.
        
        Args:
            graph (Graph): The graph to search.
            start_value (str | int): The value of the starting node.
            goal_value (str | int): The value of the goal node.
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
        stack.enqueue(start_value)

        while not stack.is_empty():
            current_value = stack.dequeue() # Get next node to explore from frontier
            if current_value == goal_value: # Check if we reached the goal
                # Reconstruct path from start to goal
                path = []
                while current_value is not None:
                    path.append(current_value)
                    current_value = parent_map.get(current_value, None)
                return True, path[::-1], visited, stack # Return reversed path i.e. from start to goal
            
            if current_value not in visited: # If we haven't visited this node yet
                visited.add(current_value) # Mark as visited
                for edge in graph.edges[current_value]: # Explore all edges from current node
                    
                    #Logic a bit funny, but has to check both directions of the edge
                    next_node = edge.to_node.value if edge.from_node.value == current_value else edge.from_node.value # Get the next node to explore
                    if next_node not in visited:
                        parent_map[next_node] = current_value
                        stack.enqueue(next_node)
        return False, [], visited, stack  # Path not found
    
    @staticmethod
    def breadth_first_search(graph: Graph, start_value: str | int, goal_value: str | int) -> tuple[bool, list, set, Queue]:
        """
        Performs a breadth-first search on the graph from start_value to goal_value.
        
        Args:
            graph (Graph): The graph to search.
            start_value (str | int): The value of the starting node.
            goal_value (str | int): The value of the goal node.
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
        queue.enqueue(start_value)

        while not queue.is_empty():
            current_value = queue.dequeue()
            if current_value == goal_value:
                # Reconstruct path from start to goal
                path = []
                while current_value is not None:
                    path.append(current_value)
                    current_value = parent_map.get(current_value, None)
                return True, path[::-1], visited, queue
            if current_value not in visited:
                visited.add(current_value)
                for edge in graph.edges[current_value]:
                    next_node = edge.to_node.value if edge.from_node.value == current_value else edge.from_node.value
                    if next_node not in visited and next_node not in queue.items:
                        parent_map[next_node] = current_value
                        queue.enqueue(next_node)
        return False, [], visited, queue
    
    @staticmethod
    def dikstra(graph: Graph, start_value: str | int, goal_value: str | int) -> tuple[bool, list, set, PriorityQueue]:
        """
        Performs Dijkstra's algorithm on the graph from start_value to goal_value.
        
        Args:
            graph (Graph): The graph to search.
            start_value (str | int): The value of the starting node.
            goal_value (str | int): The value of the goal node.
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
        distances = {node.value: float('inf') for node in graph.nodes.values()}
        distances[start_value] = 0
        priority_queue.enqueue(start_value, 0)

        while not priority_queue.is_empty():
            current_value = priority_queue.dequeue()
            if current_value == goal_value:
                # Reconstruct path from start to goal
                path = []
                while current_value is not None:
                    path.append(current_value)
                    current_value = parent_map.get(current_value, None)
                return True, path[::-1], visited, priority_queue
            
            if current_value not in visited:
                visited.add(current_value)
                for edge in graph.edges[current_value]:
                    next_node = edge.to_node.value if edge.from_node.value == current_value else edge.from_node.value
                    new_distance = distances[current_value] + edge.weight
                    if new_distance < distances[next_node]:
                        distances[next_node] = new_distance
                        parent_map[next_node] = current_value
                        priority_queue.enqueue(next_node, new_distance)
        
        return False, [], visited, priority_queue
    
    @staticmethod
    def greedy_first_search(graph: Graph, start_value: str | int, goal_value: str | int) -> tuple[bool, list, set, PriorityQueue]:
        """
        Performs a greedy first search on the graph from start_value to goal_value.
        
        Args:
            graph (Graph): The graph to search.
            start_value (str | int): The value of the starting node.
            goal_value (str | int): The value of the goal node.
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
        priority_queue.enqueue(start_value, graph.nodes[start_value].heuristic)

        while not priority_queue.is_empty():
            current_value = priority_queue.dequeue()
            if current_value == goal_value:
                # Reconstruct path from start to goal
                path = []
                while current_value is not None:
                    path.append(current_value)
                    current_value = parent_map.get(current_value, None)
                return True, path[::-1], visited, priority_queue
            
            if current_value not in visited:
                visited.add(current_value)
                for edge in graph.edges[current_value]:
                    next_node = edge.to_node.value if edge.from_node.value == current_value else edge.from_node.value
                    if next_node not in visited:
                        parent_map[next_node] = current_value
                        priority_queue.enqueue(next_node, graph.nodes[next_node].heuristic)
        
        return False, [], visited, priority_queue
    
    @staticmethod
    def a_star(graph: Graph, start_value: str | int, goal_value: str | int) -> tuple[bool, list, set, PriorityQueue]:
        """
        Performs A* search on the graph from start_value to goal_value.
        
        Args:
            graph (Graph): The graph to search.
            start_value (str | int): The value of the starting node.
            goal_value (str | int): The value of the goal node.
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
        g_scores = {node.value: float('inf') for node in graph.nodes.values()}
        g_scores[start_value] = 0
        f_scores = {node.value: float('inf') for node in graph.nodes.values()}
        f_scores[start_value] = graph.nodes[start_value].heuristic
        priority_queue.enqueue(start_value, f_scores[start_value])

        while not priority_queue.is_empty():
            current_value = priority_queue.dequeue()
            if current_value == goal_value:
                # Reconstruct path from start to goal
                path = []
                while current_value is not None:
                    path.append(current_value)
                    current_value = parent_map.get(current_value, None)
                return True, path[::-1], visited, priority_queue
            
            if current_value not in visited:
                visited.add(current_value)
                for edge in graph.edges[current_value]:
                    next_node = edge.to_node.value if edge.from_node.value == current_value else edge.from_node.value
                    tentative_g_score = g_scores[current_value] + edge.weight
                    
                    if tentative_g_score < g_scores[next_node]:
                        parent_map[next_node] = current_value
                        g_scores[next_node] = tentative_g_score
                        f_scores[next_node] = tentative_g_score + graph.nodes[next_node].heuristic
                        priority_queue.enqueue(next_node, f_scores[next_node])
        
        return False, [], visited, priority_queue

    @staticmethod
    def evaluate_search_algorithm(algorithm, graph: Graph, start_value: str | int, goal_value: str | int):
        """
        Evaluates a search algorithm on the graph. And logs the formattet results.
        
        Args:
            algorithm (function): The search algorithm to evaluate.
            graph (Graph): The graph to search.
            start_value (str | int): The value of the starting node.
            goal_value (str | int): The value of the goal node.
        Returns:
            None
        """
        
        found, path, visited, frontier = algorithm(graph, start_value, goal_value)
        print()
        logging.info(f"Search Algorithm: {algorithm.__name__}")
        logging.info(f"Start Node: {start_value}, Goal Node: {goal_value}")
        logging.info(f"Goal Found: {found}")
        logging.info(f"Path: {' -> '.join(path) if path else 'No path found'}")
        logging.info(f"Visited Nodes: {len(visited)}")
        logging.info(f"Frontier Size: {len(frontier.items)}")



if __name__ == '__main__':
    # FILEPATH = 'src/examples/figure2_graph.csv'
    # graph = Graph()
    # graph.load(FILEPATH)
    # logging.info(f'Graph loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges.')

    # graph.vizualize()
    # logging.info('Graph visualization complete.')

    FILEPATH = 'src/examples/figure4_graph.csv'
    graph = Graph()
    graph.load(FILEPATH)
    logging.info(f'Graph loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges.')

    graph.vizualize() # Does not work after Python 3.12, distutils are deprecated
    logging.info('Graph visualization complete.')
    
    search.evaluate_search_algorithm(search.depth_first_search, graph, 's0', 's23')
    
    search.evaluate_search_algorithm(search.breadth_first_search, graph, 's0', 's23')
    
    search.evaluate_search_algorithm(search.dikstra, graph, 's0', 's23')
    
    search.evaluate_search_algorithm(search.greedy_first_search, graph, 's0', 's23')
    
    search.evaluate_search_algorithm(search.a_star, graph, 's0', 's23')


