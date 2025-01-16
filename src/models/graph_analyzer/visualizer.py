import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def plot_graph(graph: nx.DiGraph, with_labels: bool = True) -> None:
    """Plot a graph using matplotlib."""
    plt.figure(figsize=(15, 10))            
    pos = nx.kamada_kawai_layout(graph)
    event_labels = nx.get_edge_attributes(graph, 'event')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=event_labels)
    nx.draw(graph, pos, with_labels=with_labels, node_color='lightblue', 
            edge_color='grey', width=2)
    plt.show()

def save_graph(graph: nx.DiGraph, filename: str | Path, with_labels: bool = True) -> None:
    """Save a graph visualization to a file."""
    plt.figure(figsize=(15, 10))            
    pos = nx.kamada_kawai_layout(graph)
    event_labels = nx.get_edge_attributes(graph, 'event')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=event_labels)
    nx.draw(graph, pos, with_labels=with_labels, node_color='lightblue', 
            edge_color='grey', width=2)
    plt.savefig(filename)
    plt.close() 