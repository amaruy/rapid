import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set, List, Dict, Union, Tuple
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec

from .utils import find_ttps, filter_files, filter_processes
from .visualizer import plot_graph, save_graph
from src.utils.logging_utils import setup_logger

class GraphAnalyzer:
    """Analyzes system events and creates graph representations for anomaly detection."""
    
    def __init__(self, source: str, df: Optional[pd.DataFrame] = None, verbose: bool = True):
        """Initialize the GraphAnalyzer.
        
        Args:
            source: Source identifier for the data
            df: Optional DataFrame containing system events
            verbose: Whether to print progress messages
        """
        self.source = source
        self.verbose = verbose
        self.logger = setup_logger('graph_analyzer', Path('artifacts/logs'))
        
        if df is not None:
            self.logger.info(f'Preprocessing data... {len(df)} logs.')
            self.df = self.preprocess(df)
            self.logger.info('Clustering objects')
            clusters = self.cluster_objects(self.df['objectData'])
            self.df['cluster'] = self.df['objectData'].map(clusters).fillna(0)
            self.logger.info(f'Creating graph from {len(self.df)} logs...')
            self.graph = self.create_graph(self.df)
            self.logger.info(f'Graph created. {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.')
            self.df_dict = self.df.set_index('uuid')[['processUUID', 'objectUUID', 'objectData', 'event',
                                                   'dataflow', 'ttp', 'cluster']].to_dict(orient='index')
            self.logger.info('GraphAnalyzer initialized.')

    def printer(self, string: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            self.logger.info(string)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input DataFrame for graph creation.
        
        Args:
            df: Input DataFrame containing system events
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        df = df.dropna(subset=['processUUID', 'objectData', 'event'])
        df = df[(df['objectData'] != 'NA') & (df['objectData'] != '<unknown>')]
        
        # Correct event types
        df.loc[df['objectType'] == 'socket', 'event'] = df.loc[
            df['objectType'] == 'socket', 'event'
        ].replace({'read': 'receive', 'write': 'send'})
        
        # Add dataflow direction
        df['dataflow'] = 'outward'
        df.loc[df['event'].isin(['read', 'receive', 'execute']), 'dataflow'] = 'inward'
        
        # Add TTPs
        df['ttp'] = df.apply(find_ttps, axis=1)
        
        # Filter important entities
        imp_files = filter_files(df)
        imp_processes = filter_processes(df)
        
        df['imp_file'] = df['objectData'].isin(imp_files)
        df['imp_object'] = df['objectUUID'].isin(imp_processes)
        df['external'] = df['objectData'].apply(
            lambda obj: not any(ip in str(obj) for ip in ['192.168.', '128.55.12.', '127.0.0.1'])
        ) & (df['objectType'] == 'socket')
        
        df['imp_object'] = df['imp_file'] | df['imp_object'] | df['external']
        df['imp_process'] = df['processUUID'].isin(imp_processes)

        return df[df['imp_process'] & df['imp_object']].copy()

    def cluster_objects(self, objects: List[str], min_samples: int = 3) -> Dict[str, int]:
        """Cluster object strings using DBSCAN on their embeddings.

        Args:
            objects: List of object identifiers to cluster
            min_samples: Minimum samples parameter for DBSCAN

        Returns:
            Mapping of object identifier to cluster label
        """

        try:
            model_path = f"checkpoints/{self.source}/word2vec.model"
            w2v_model = Word2Vec.load(model_path)
        except Exception as e:
            self.logger.warning(f"Failed loading embeddings from {model_path}: {e}")
            return {}

        unique_obj_data = np.unique(objects)
        embeddings: List[np.ndarray] = []
        obj_names: List[str] = []

        for obj in unique_obj_data:
            if obj in w2v_model.wv:
                embeddings.append(w2v_model.wv[obj])
                obj_names.append(obj)

        if not embeddings:
            return {}

        embeddings_array = np.array(embeddings)

        tsne = TSNE(n_components=2, random_state=0)
        embeddings_2d = tsne.fit_transform(embeddings_array)

        eps = self._estimate_eps(embeddings_2d, min_samples)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings_2d)
        labels = dbscan.labels_

        return dict(zip(obj_names, labels))

    def _estimate_eps(self, points: np.ndarray, min_samples: int) -> float:
        """Estimate DBSCAN epsilon using the elbow method.

        Args:
            points: 2D array of points
            min_samples: Number of neighbors to consider

        Returns:
            Estimated epsilon value
        """
        nn = NearestNeighbors(n_neighbors=min_samples).fit(points)
        distances, _ = nn.kneighbors(points)
        k_distances = np.sort(distances[:, min_samples - 1])
        differences = np.diff(k_distances)
        elbow_index = np.argmax(differences)
        return float(k_distances[elbow_index])

    @staticmethod
    def add_node(G: nx.DiGraph, identifier: str, timestamp: int, node_type: str) -> None:
        """Add a node to the graph if it doesn't exist."""
        if not G.has_node(identifier):
            G.add_node(identifier, timestamp=timestamp, type=node_type)

    @staticmethod
    def add_edge(G: nx.DiGraph, source: str, target: str, timestamp: int, 
                event: str, ttp: str, cluster: int) -> None:
        """Add an edge to the graph."""
        G.add_edge(source, target, timestamp=timestamp, event=event, ttp=ttp, cluster=cluster)

    def create_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Create a directed graph from the preprocessed DataFrame.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            NetworkX DiGraph
        """
        df = df.sort_values('timestamp')
        G = nx.DiGraph()
        nodes = set()
        
        for row in df.itertuples(index=False):
            subj = row.processUUID
            obj = row.objectUUID if row.event == 'fork' else row.objectData
            node_type = row.objectType
            event = row.event
            
            if event == 'receive':
                nodes.add(obj)
                
            if row.dataflow == 'outward':
                if subj in nodes:
                    nodes.add(obj)
                    self.add_node(G, subj, row.timestamp, 'process')
                    self.add_node(G, obj, row.timestamp, node_type)
                    self.add_edge(G, subj, obj, row.timestamp, row.event, row.ttp, row.cluster)
                elif row.event == 'fork':
                    self.add_node(G, subj, row.timestamp, 'process')
                    self.add_node(G, obj, row.timestamp, node_type)
                    self.add_edge(G, subj, obj, row.timestamp, row.event, row.ttp, row.cluster)
            elif row.dataflow == 'inward':
                if obj in nodes:
                    nodes.add(subj)
                    self.add_node(G, obj, row.timestamp, node_type)
                    self.add_node(G, subj, row.timestamp, 'process')
                    self.add_edge(G, obj, subj, row.timestamp, row.event, row.ttp, row.cluster)
                elif subj in nodes:
                    self.add_node(G, obj, row.timestamp, node_type)
                    self.add_node(G, subj, row.timestamp, 'process')
                    self.add_edge(G, obj, subj, row.timestamp, row.event, row.ttp, row.cluster)
        
        # Drop system nodes
        system_entities = [
            'C:\\Windows\\', 'C:\\Program Files (x86)\\', 'C:\\Program Files\\',
            'C:\\windows\\', 'C:\\Users\\Administrator\\AppData\\', 'C:\\ProgramData\\',
            'C:\\Users\\ADMINI~1\\AppData\\', 'C:\\program files\\', 'C:\\PROGRAMDATA\\',
            '.ini', 'GoogleUpdate', 'unknown'
        ]
        system_nodes = [node for node in G.nodes() if any(path in str(node) for path in system_entities)]
        G.remove_nodes_from(system_nodes)
        
        return G

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors of a node in the graph."""
        return nx.ancestors(self.graph, node)
    
    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node in the graph."""
        return nx.descendants(self.graph, node)

    def get_graph(self, uuid: str, threshold: float = 0.01) -> nx.DiGraph:
        """Get a subgraph for a specific event UUID.
        
        Args:
            uuid: Event UUID
            threshold: Score threshold for filtering edges
            
        Returns:
            Subgraph as NetworkX DiGraph
        """
        if uuid not in self.df_dict:
            return nx.DiGraph()
        
        event_data = self.df_dict[uuid]
        subject = event_data['processUUID']
        object = event_data['objectUUID'] if event_data['event'] == 'fork' else event_data['objectData']
        direction = event_data['dataflow']
        cluster = event_data['cluster']
        
        if subject not in self.graph or object not in self.graph:
            return nx.DiGraph()
        
        if direction == 'outward':
            ancestors = self.get_ancestors(subject)
            descendants = self.get_descendants(object)
        else:
            ancestors = self.get_ancestors(object)
            descendants = self.get_descendants(subject)
        
        nodes = ancestors.union(descendants).union({subject, object})
        subgraph = self.graph.subgraph(nodes).copy()
        
        # Filter edges
        edges_to_remove = [
            (u, v) for u, v, attrs in subgraph.edges(data=True)
            if attrs.get('cluster') != cluster or attrs.get('score', 0) < threshold
        ]
        
        subgraph.remove_edges_from(edges_to_remove)
        subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
        
        return subgraph

    def get_score(self, uuid: Optional[str] = None, graph: Optional[nx.DiGraph] = None) -> float:
        """Calculate anomaly score for a graph.
        
        Args:
            uuid: Optional event UUID to get graph from
            graph: Optional graph to score directly
            
        Returns:
            Anomaly score
        """
        if graph is None and uuid:
            graph = self.get_graph(uuid)
            
        edge_scores = nx.get_edge_attributes(graph, 'score')
        edge_AS = sum(edge_scores.values()) / len(edge_scores) if edge_scores else 0
        
        edge_labels = nx.get_edge_attributes(graph, 'ttp')
        labelled_events = {edge_labels[edge] for edge in edge_labels if edge_labels[edge]}
        edge_TS = len(labelled_events) / 7 if labelled_events else 0
        
        return edge_AS + edge_TS

    def map_scores(self, anomalies: Dict[str, float]) -> None:
        """Map anomaly scores to graph edges.
        
        Args:
            anomalies: Dictionary mapping UUIDs to anomaly scores
        """
        self.df['score'] = self.df['uuid'].map(anomalies)
        self.df['score'] = self.df['score'].fillna(0)
        
        self.df['object'] = np.where(
            self.df['event'] == 'fork',
            self.df['objectUUID'],
            self.df['objectData']
        )
        
        self.df = self.df.sort_values('score', ascending=False)
        self.df = self.df.drop_duplicates(
            subset=['processUUID', 'object', 'dataflow'],
            keep='first'
        )
        
        for row in self.df.itertuples(index=False):
            if row.dataflow == 'outward':
                if self.graph.has_edge(row.processUUID, row.object):
                    self.graph.edges[row.processUUID, row.object]['score'] = row.score
            elif row.dataflow == 'inward':
                if self.graph.has_edge(row.object, row.processUUID):
                    self.graph.edges[row.object, row.processUUID]['score'] = row.score

    def plot(self, graph: Optional[nx.DiGraph] = None) -> None:
        """Plot the graph or a subgraph."""
        plot_graph(graph if graph is not None else self.graph)

    def save(self, graph: nx.DiGraph, filename: Union[str, Path]) -> None:
        """Save the graph visualization to a file."""
        save_graph(graph, filename)

    def merge_alerts(
        self, alerts: List[Tuple[str, nx.DiGraph, float]]
    ) -> List[Tuple[str, nx.DiGraph, float]]:
        """Merge alerts whose graphs share nodes.

        Args:
            alerts: List of ``(trigger_uuid, alert_graph, score)`` tuples.

        Returns:
            List of merged alert tuples.
        """

        merged: List[Tuple[str, nx.DiGraph, float]] = []

        for uuid, graph, _ in alerts:
            merge_idx: Optional[int] = None
            for idx, (_, m_graph, _) in enumerate(merged):
                if set(graph.nodes()).intersection(m_graph.nodes()):
                    merge_idx = idx
                    break

            if merge_idx is not None:
                m_uuid, m_graph, _ = merged[merge_idx]
                combined = nx.compose(m_graph, graph)
                merged_score = self.get_score(graph=combined)
                merged[merge_idx] = (m_uuid, combined, merged_score)
            else:
                merged.append((uuid, graph, self.get_score(graph=graph)))

        return merged
