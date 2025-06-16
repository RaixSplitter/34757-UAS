import csv
import math

def compute_weight(pos1, pos2):
    """Calculate Euclidean distance between two points."""
    return round(math.hypot(pos2[0] - pos1[0], pos2[1] - pos1[1]), 2)

def export_edges_to_csv(nodes, edges, filename='maze_graph.csv'):
    """Export all edges as comma-separated values with label 'E' and adjusted coordinates."""
    written = set()

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')  # <-- changed to comma

        for src, neighbors in edges.items():
            for dst in neighbors:
                edge_key = tuple(sorted([src, dst]))
                if edge_key not in written:
                    weight = compute_weight(nodes[src], nodes[dst])
                    x, y = nodes[dst]
                    writer.writerow(['E', src, dst, weight, x - 1, y - 1])  # shifted coords
                    written.add(edge_key)

    print(f"[✓] Exported {len(written)} edges to {filename} using commas")



def export_edges_to_csv_bidirectional(nodes, edges, filename='maze_graph.csv'):
    """Exports undirected edges as two directed lines: E,from,to,weight"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        
        for src, neighbors in edges.items():
            for dst in neighbors:
                weight = compute_weight(nodes[src], nodes[dst])
                writer.writerow(['E', src, dst, int(weight)])  # Assuming weight is always 1.0

    print(f"[✓] Exported directed edges to {filename}")
