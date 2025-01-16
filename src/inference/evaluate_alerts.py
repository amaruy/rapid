from pathlib import Path
import yaml
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple
import logging

from src.utils.logging_utils import setup_logger

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_ground_truth(
    data_dir: Path,
    config: dict,
    logger: logging.Logger
) -> Tuple[List[str], List[str]]:
    """Load ground truth malicious objects and subjects.
    
    Args:
        data_dir: Data directory path
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (malicious objects list, malicious subjects list)
    """
    logger.info("Loading ground truth data")
    
    # Load malicious objects
    objects_path = data_dir / config['malicious_objects_file']
    with open(objects_path) as f:
        malicious_objects = f.read().splitlines()
    logger.info(f"Loaded {len(malicious_objects)} malicious objects")
    
    # Load malicious subjects
    subjects_path = data_dir / config['malicious_subjects_file']
    with open(subjects_path) as f:
        malicious_subjects = f.read().splitlines()
    logger.info(f"Loaded {len(malicious_subjects)} malicious subjects")
    
    return malicious_objects, malicious_subjects

def get_ground_truth_edges(
    df: pd.DataFrame,
    malicious_entities: List[str],
    exclude_events: List[str],
    logger: logging.Logger
) -> Set[Tuple[str, str]]:
    """Extract ground truth edges from events.
    
    Args:
        df: DataFrame with system events
        malicious_entities: List of malicious objects and subjects
        exclude_events: List of events to exclude
        logger: Logger instance
        
    Returns:
        Set of ground truth edges as (source, target) tuples
    """
    logger.info("Extracting ground truth edges")
    
    # Filter events involving malicious entities
    malicious_events = df[
        df['processUUID'].isin(malicious_entities) & 
        ((df['objectData'].isin(malicious_entities)) | 
         (df['objectUUID'].isin(malicious_entities)))
    ]
    
    # Exclude specified events
    if exclude_events:
        malicious_events = malicious_events[~malicious_events['event'].isin(exclude_events)]
    
    # Get unique edges
    key_events = malicious_events.drop_duplicates(
        subset=['processUUID', 'objectUUID', 'objectData', 'event']
    )
    
    # Convert events to edges
    edges = set()
    for _, event in key_events.iterrows():
        # For fork events, connect process to forked process
        if event['event'] == 'fork':
            edges.add((event['processUUID'], event['objectUUID']))
        # For other events, connect process to object
        else:
            # Process is always the source for simplicity
            edges.add((event['processUUID'], event['objectData']))
    
    logger.info(f"Found {len(edges)} ground truth edges")
    return edges

def evaluate_edges(
    detected_edges: Set[Tuple[str, str]],
    ground_truth_edges: Set[Tuple[str, str]],
    total_events: int,
    logger: logging.Logger
) -> Tuple[Dict[str, float], Dict[str, Set[Tuple[str, str]]]]:
    """Evaluate edge detection performance.
    
    Args:
        detected_edges: Set of detected edges
        ground_truth_edges: Set of ground truth edges
        total_events: Total number of unique events
        logger: Logger instance
        
    Returns:
        Tuple of (metrics dict, edge sets dict)
    """
    logger.info("Evaluating edge detection performance")
    
    # Calculate true positives, false positives, and false negatives
    true_positives = detected_edges.intersection(ground_truth_edges)
    false_positives = detected_edges - ground_truth_edges
    false_negatives = ground_truth_edges - detected_edges
    
    # Calculate metrics
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    tn = total_events - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }
    
    edge_sets = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"FPR: {fpr:.4f}")
    
    return metrics, edge_sets

def save_results(
    metrics: Dict[str, float],
    edge_sets: Dict[str, Set[Tuple[str, str]]],
    output_dir: Path,
    config: dict,
    logger: logging.Logger
) -> None:
    """Save evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        edge_sets: Dictionary of edge sets
        output_dir: Output directory path
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Saving evaluation results")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / config['metrics_file']
    with open(metrics_path, 'w') as f:
        f.write("Detection Performance Metrics\n")
        f.write("===========================\n\n")
        
        f.write("Edge-Level Metrics:\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"False Positive Rate: {metrics['fpr']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"True Positives: {metrics['true_positives']}\n")
        f.write(f"False Positives: {metrics['false_positives']}\n")
        f.write(f"False Negatives: {metrics['false_negatives']}\n")
        f.write(f"True Negatives: {metrics['true_negatives']}\n")
    
    # Save edge sets
    for set_name, edges in edge_sets.items():
        file_path = output_dir / config[f'{set_name}_file']
        with open(file_path, 'w') as f:
            for edge in sorted(edges):
                f.write(f"{edge[0]}, {edge[1]}\n")
    
    logger.info(f"Results saved to {output_dir}")

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    
    # Load configs
    base_config = load_config(root_dir / "config" / "config.yaml")
    infer_config = load_config(root_dir / "config" / "inference_config.yaml")
    
    # Setup logging
    logger = setup_logger(
        "evaluation",
        root_dir / base_config['paths']['logs_dir']
    )
    logger.info("Starting evaluation")
    
    try:
        # Load test data
        data_path = root_dir / base_config['paths']['data_dir'] / "test_logs.pkl"
        logger.info(f"Loading test data from {data_path}")
        df = pd.read_pickle(data_path)
        
        # Load ground truth
        malicious_objects, malicious_subjects = load_ground_truth(
            root_dir / base_config['paths']['data_dir'],
            infer_config['evaluation'],
            logger
        )
        malicious_entities = malicious_objects + malicious_subjects
        
        # Get ground truth edges
        ground_truth_edges = get_ground_truth_edges(
            df,
            malicious_entities,
            infer_config['evaluation']['exclude_events'],
            logger
        )
        
        # Load and combine alert graphs
        alerts_dir = root_dir / infer_config['graph_analysis']['alerts_dir']
        graphs_dir = alerts_dir / infer_config['graph_analysis']['alert_graphs_dir']
        logger.info(f"Loading alert graphs from {graphs_dir}")
        
        detected_edges = set()
        for graph_file in graphs_dir.glob("alert_*.pkl"):
            try:
                with open(graph_file, 'rb') as f:
                    alert_graph = pickle.load(f)
                    detected_edges.update(alert_graph.edges())
            except Exception as e:
                logger.warning(f"Failed to load graph from {graph_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(detected_edges)} total detected edges")
        
        # Calculate total unique events for true negative calculation
        total_events = len(df.drop_duplicates(
            subset=['processUUID', 'objectUUID', 'objectData', 'event']
        ))
        
        # Evaluate detection performance
        metrics, edge_sets = evaluate_edges(
            detected_edges,
            ground_truth_edges,
            total_events,
            logger
        )
        
        # Save results
        output_dir = root_dir / infer_config['evaluation']['output_dir']
        save_results(
            metrics,
            edge_sets,
            output_dir,
            infer_config['evaluation'],
            logger
        )
        
        logger.info("Evaluation complete")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main() 