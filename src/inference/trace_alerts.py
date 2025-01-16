from pathlib import Path
import yaml
import pickle
import pandas as pd
import logging
import networkx as nx
from typing import List, Dict, Optional, Tuple

from src.utils.logging_utils import setup_logger
from src.models.graph_analyzer.analyzer import GraphAnalyzer

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_analyzer(
    df: pd.DataFrame,
    anomaly_scores: Dict[str, float],
    config: dict,
    logger: logging.Logger
) -> GraphAnalyzer:
    """Setup and initialize the graph analyzer.
    
    Args:
        df: Input DataFrame with system events
        anomaly_scores: Dictionary mapping UUIDs to anomaly scores
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Initialized GraphAnalyzer
    """
    logger.info("Initializing graph analyzer")
    
    # Filter out system paths and processes
    logger.info("Filtering system events")
    try:
        # Apply system path exclusions
        mask = ~df['objectData'].apply(
            lambda x: any(path in str(x) for path in config['exclude_system_paths'])
        )
        logger.info(f"Filtered {len(df) - mask.sum()} events with system paths")
        
        # Apply extension exclusions
        mask &= ~df['objectData'].apply(
            lambda x: any(str(x).lower().endswith(ext) for ext in config['exclude_extensions'])
        )
        logger.info(f"Remaining events after extension filtering: {mask.sum()}")
        
        # Apply process exclusions
        mask &= ~df['processName'].isin(config['exclude_processes'])
        logger.info(f"Final events after process filtering: {mask.sum()}")
        
        df = df[mask].copy()
        
    except Exception as e:
        logger.warning(f"Error during event filtering: {e}")
        logger.warning("Proceeding with unfiltered data")
    
    try:
        analyzer = GraphAnalyzer(source=config['source'], df=df, verbose=False)
        
        logger.info("Mapping anomaly scores to graph")
        analyzer.map_scores(anomaly_scores)
        
        logger.info(f"Graph constructed with {len(analyzer.graph.nodes)} nodes and {len(analyzer.graph.edges)} edges")
        return analyzer
        
    except Exception as e:
        logger.error(f"Failed to initialize graph analyzer: {e}")
        raise

def trace_alerts(
    analyzer: GraphAnalyzer,
    config: dict,
    logger: logging.Logger
) -> List[Tuple[str, nx.DiGraph, float]]:
    """Trace and analyze alerts in the graph.
    
    Args:
        analyzer: Initialized GraphAnalyzer
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        List of tuples containing (trigger_uuid, alert_graph, alert_score)
    """
    logger.info("Starting alert tracing")
    
    try:
        # Get top anomalous events
        filtered = analyzer.df.sort_values('score', ascending=False).drop_duplicates(
            subset=['processUUID', 'objectData', 'dataflow']
        )
        top_events = filtered[filtered['score'] >= config['alert_threshold']].head(
            config['top_n_alerts']
        )
        
        logger.info(f"Found {len(top_events)} events above threshold {config['alert_threshold']}")
        
        # Trace alerts
        alerts = []
        for _, event in top_events.iterrows():
            try:
                # Get alert graph
                alert = analyzer.get_graph(event['uuid'], threshold=config['event_threshold'])
                if alert is None:
                    continue
                    
                # Score the alert
                score = analyzer.get_score(graph=alert)
                if score > config['alert_threshold']:
                    alerts.append((event['uuid'], alert, score))
                    
            except Exception as e:
                logger.warning(f"Failed to process event {event['uuid']}: {e}")
                continue
                
        logger.info(f"Found {len(alerts)} significant alerts")
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to trace alerts: {e}")
        raise

def save_alert_details(
    alerts: List[Tuple[str, nx.DiGraph, float]],
    analyzer: GraphAnalyzer,
    config: dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Save alert graphs and details.
    
    Args:
        alerts: List of (trigger_uuid, alert_graph, alert_score) tuples
        analyzer: GraphAnalyzer instance for graph operations
        config: Configuration dictionary
        output_dir: Output directory path
        logger: Logger instance
    """
    try:
        graphs_dir = output_dir / config['alert_graphs_dir']
        details_dir = output_dir / config['alert_details_dir']
        
        # Create output directories
        graphs_dir.mkdir(parents=True, exist_ok=True)
        details_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving alert details")
        
        # Save summary file
        with open(output_dir / "alert_summary.txt", 'w') as f:
            f.write(f"Analysis Summary\n")
            f.write(f"================\n")
            f.write(f"Source: {config['source']}\n")
            f.write(f"Total alerts found: {len(alerts)}\n")
            f.write(f"Alert threshold: {config['alert_threshold']}\n")
            f.write(f"Event threshold: {config['event_threshold']}\n\n")
            
            f.write(f"Alert Details\n")
            f.write(f"============\n")
            for i, (uuid, graph, score) in enumerate(alerts):
                event = analyzer.df[analyzer.df['uuid'] == uuid].iloc[0]
                f.write(f"Alert {i}:\n")
                f.write(f"Trigger UUID: {uuid}\n")
                f.write(f"Process: {event['processName']}\n")
                f.write(f"Event: {event['event']}\n")
                f.write(f"Object: {event['objectData']}\n")
                f.write(f"Score: {score:.4f}\n")
                f.write(f"Size: {len(graph.edges)} edges, {len(graph.nodes)} nodes\n")
                f.write(f"Graph saved: {len(graph.edges) <= config['max_graph_size'] and score >= config['min_graph_score']}\n\n")
                
        # Save individual alert details and graphs
        for i, (uuid, graph, score) in enumerate(alerts):
            if graph is None:
                continue
                
            # Save graph pickle
            graph_pickle_path = graphs_dir / f"alert_{i}.pkl"
            with open(graph_pickle_path, 'wb') as f:
                pickle.dump(graph, f)
                
            # Save graph visualization if not too large
            if len(graph.edges) <= config['max_graph_size'] and score >= config['min_graph_score']:
                try:
                    graph_path = graphs_dir / f"alert_{i}.png"
                    analyzer.save_graph(graph, str(graph_path))
                except Exception as e:
                    logger.error(f"Failed to save graph visualization for alert {i}: {e}")
                
            # Save graph details
            details_path = details_dir / f"alert_{i}.txt"
            with open(details_path, 'w') as f:
                # Write basic info
                event = analyzer.df[analyzer.df['uuid'] == uuid].iloc[0]
                f.write(f"Alert {i}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Trigger Event:\n")
                f.write(f"  UUID: {uuid}\n")
                f.write(f"  Process: {event['processName']}\n")
                f.write(f"  Event: {event['event']}\n")
                f.write(f"  Object Type: {event['objectType']}\n")
                f.write(f"  Object: {event['objectData']}\n")
                f.write(f"  Timestamp: {event.get('timestamp', 'unknown')}\n")
                f.write(f"Alert Score: {score:.4f}\n")
                f.write(f"Graph Size: {len(graph.nodes)} nodes, {len(graph.edges)} edges\n\n")
                
                # Write node details
                f.write(f"Node Details:\n")
                f.write(f"{'-'*50}\n")
                for node in sorted(graph.nodes()):
                    node_data = graph.nodes[node]
                    f.write(f"Node: {node}\n")
                    f.write(f"Type: {node_data.get('type', 'unknown')}\n")
                    f.write(f"First seen: {node_data.get('timestamp', 'unknown')}\n\n")
                
                # Write edge details
                f.write(f"Edge Details:\n")
                f.write(f"{'-'*50}\n")
                for edge in sorted(graph.edges(data=True)):
                    f.write(f"Source: {edge[0]}\n")
                    f.write(f"Target: {edge[1]}\n")
                    f.write("Attributes:\n")
                    for key, value in sorted(edge[2].items()):
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
        
        logger.info(f"Saved details for {len(alerts)} alerts")
        
    except Exception as e:
        logger.error(f"Failed to save alert details: {e}")
        raise

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    
    # Load configs
    base_config = load_config(root_dir / "config" / "config.yaml")
    infer_config = load_config(root_dir / "config" / "inference_config.yaml")
    
    # Setup logging
    logger = setup_logger(
        "graph_analysis",
        root_dir / base_config['paths']['logs_dir']
    )
    logger.info("Starting graph analysis")
    
    try:
        # Load test data
        data_path = root_dir / base_config['paths']['data_dir'] / "test_logs.pkl"
        logger.info(f"Loading test data from {data_path}")
        df = pd.read_pickle(data_path)
        
        # Load anomaly scores
        scores_path = root_dir / infer_config['anomaly_detection']['output_dir'] / infer_config['anomaly_detection']['scores_filename']
        logger.info(f"Loading anomaly scores from {scores_path}")
        with open(scores_path, 'rb') as f:
            anomaly_scores = pickle.load(f)
        
        # Setup analyzer
        config = infer_config['graph_analysis']
        analyzer = setup_analyzer(df, anomaly_scores, config, logger)
        
        # Save analyzer
        graph_dir = root_dir / config['graph_dir']
        graph_dir.mkdir(parents=True, exist_ok=True)
        analyzer_path = graph_dir / config['analyzer_filename']
        with open(analyzer_path, 'wb') as f:
            pickle.dump(analyzer, f)
        logger.info(f"Saved analyzer to {analyzer_path}")
        
        # Trace and analyze alerts
        alerts = trace_alerts(analyzer, config, logger)
        
        # Save results
        alerts_dir = root_dir / config['alerts_dir']
        alerts_dir.mkdir(parents=True, exist_ok=True)
        save_alert_details(alerts, analyzer, config, alerts_dir, logger)
        logger.info("Completed graph analysis")
        
    except Exception as e:
        logger.error(f"Graph analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 