"""State serialization for Graphenery application."""
import json
import gzip
import datetime
from typing import Dict, List, Tuple, Any
import networkx as nx
import numpy as np


class StateSerializer:
    """Handle serialization and deserialization of application state."""
    
    VERSION = "1.0"
    
    @staticmethod
    def serialize_graph(graph: nx.Graph, positions: Dict, colors: Dict) -> Dict:
        """
        Serialize NetworkX graph with positions and colors.
        
        Args:
            graph: NetworkX graph
            positions: Node position dictionary
            colors: Node color dictionary
            
        Returns:
            Dictionary with serialized graph data
        """
        # Convert nodes to list (they're tuples)
        nodes = [list(node) for node in graph.nodes()]
        
        # Convert edges with weights
        edges = []
        for u, v, data in graph.edges(data=True):
            edge_dict = {
                'source': list(u),
                'target': list(v),
                'weight': data.get('weight', 1.0)
            }
            edges.append(edge_dict)
        
        # Convert positions (numpy arrays to lists)
        positions_serialized = {}
        for node, pos in positions.items():
            node_key = str(node)  # Tuple to string for JSON
            if isinstance(pos, np.ndarray):
                positions_serialized[node_key] = pos.tolist()
            else:
                positions_serialized[node_key] = list(pos)
        
        # Convert colors
        colors_serialized = {str(node): color for node, color in colors.items()}
        
        return {
            'nodes': nodes,
            'edges': edges,
            'positions': positions_serialized,
            'colors': colors_serialized
        }
    
    @staticmethod
    def deserialize_graph(data: Dict) -> Tuple[nx.Graph, Dict, Dict]:
        """
        Deserialize graph from dictionary.
        
        Args:
            data: Dictionary with graph data
            
        Returns:
            Tuple of (graph, positions, colors)
        """
        # Create graph
        graph = nx.Graph()
        
        # Add nodes (convert lists back to tuples)
        nodes = [tuple(node) for node in data['nodes']]
        graph.add_nodes_from(nodes)
        
        # Add edges with weights
        for edge_dict in data['edges']:
            source = tuple(edge_dict['source'])
            target = tuple(edge_dict['target'])
            weight = edge_dict['weight']
            graph.add_edge(source, target, weight=weight)
        
        # Convert positions back
        positions = {}
        for node_str, pos_list in data['positions'].items():
            node = eval(node_str)  # String back to tuple
            positions[node] = np.array(pos_list)
        
        # Convert colors back
        colors = {}
        for node_str, color in data['colors'].items():
            node = eval(node_str)
            colors[node] = color
        
        return graph, positions, colors
    
    @staticmethod
    def serialize_operation_history(history) -> Dict:
        """
        Serialize operation history.
        
        Args:
            history: OperationHistory object
            
        Returns:
            Dictionary with operation history data
        """
        return {
            'operations': history.operations,
            'current_index': history.current_index
        }
    
    @staticmethod
    def serialize_lattice_config(config) -> Dict:
        """
        Serialize lattice configuration.
        
        Args:
            config: Configuration object or dict
            
        Returns:
            Dictionary with configuration
        """
        if hasattr(config, '__dict__'):
            # It's an object, convert to dict
            from dataclasses import asdict
            try:
                return asdict(config)
            except:
                # Fallback: manual conversion
                return {
                    'bond_length': getattr(config, 'bond_length', 1.0),
                    'num_units': getattr(config, 'num_units', 7),
                    'boundary': getattr(config, 'boundary', 'bearded'),
                    'periodic_bc': getattr(config, 'periodic_bc', False),
                }
        else:
            # Already a dict
            return config
    
    @staticmethod
    def serialize_polygons(polygon_selectors: List) -> List[Dict]:
        """
        Serialize polygon selectors for visual restoration.
        
        Args:
            polygon_selectors: List of PolygonSelector objects
            
        Returns:
            List of polygon dictionaries
        """
        polygons = []
        for ps in polygon_selectors:
            if hasattr(ps, 'vertices') and hasattr(ps, 'is_complete') and ps.is_complete:
                polygons.append({
                    'vertices': ps.vertices,
                    'color': ps.color,
                    'linewidth': ps.linewidth
                })
        return polygons
    
    @staticmethod
    def save_state(filepath: str, state_dict: Dict, compress: bool = False):
        """
        Save state dictionary to file.
        
        Args:
            filepath: Path to save file
            state_dict: State dictionary
            compress: Whether to compress with gzip
        """
        # Add version and timestamp
        state_dict['version'] = StateSerializer.VERSION
        state_dict['timestamp'] = datetime.datetime.now().isoformat()
        
        # Convert to JSON
        json_str = json.dumps(state_dict, indent=2)
        
        if compress:
            # Save compressed
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            # Save uncompressed
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
    
    @staticmethod
    def load_state(filepath: str) -> Dict:
        """
        Load state dictionary from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            State dictionary
            
        Raises:
            ValueError: If version is incompatible
        """
        # Try compressed first
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                json_str = f.read()
        except:
            # Try uncompressed
            with open(filepath, 'r', encoding='utf-8') as f:
                json_str = f.read()
        
        state_dict = json.loads(json_str)
        
        # Check version
        file_version = state_dict.get('version', '0.0')
        if file_version != StateSerializer.VERSION:
            print(f"Warning: File version {file_version} differs from current version {StateSerializer.VERSION}")
            # Could add migration logic here
        
        return state_dict


class WorkflowExporter:
    """Export and import operation workflows."""
    
    @staticmethod
    def export_workflow(operations: List[Dict], filepath: str, 
                       metadata: Dict = None):
        """
        Export operations as reusable workflow.
        
        Args:
            operations: List of operation dictionaries
            filepath: Path to save workflow
            metadata: Optional metadata (description, author, etc.)
        """
        workflow = {
            'version': StateSerializer.VERSION,
            'timestamp': datetime.datetime.now().isoformat(),
            'operations': operations,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2)
    
    @staticmethod
    def import_workflow(filepath: str) -> Tuple[List[Dict], Dict]:
        """
        Import workflow from file.
        
        Args:
            filepath: Path to workflow file
            
        Returns:
            Tuple of (operations list, metadata dict)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        operations = workflow.get('operations', [])
        metadata = workflow.get('metadata', {})
        
        return operations, metadata


class AutoSaveManager:
    """Manage automatic saving of sessions."""
    
    def __init__(self, interval_minutes: int = 5, max_backups: int = 3):
        """
        Initialize auto-save manager.
        
        Args:
            interval_minutes: Minutes between auto-saves
            max_backups: Maximum number of auto-save files to keep
        """
        self.interval_minutes = interval_minutes
        self.max_backups = max_backups
        self.last_save_time = None
        self.auto_save_enabled = False
    
    def should_auto_save(self) -> bool:
        """Check if it's time for auto-save."""
        if not self.auto_save_enabled:
            return False
        
        if self.last_save_time is None:
            return True
        
        elapsed = datetime.datetime.now() - self.last_save_time
        return elapsed.total_seconds() > (self.interval_minutes * 60)
    
    def get_auto_save_path(self, base_dir: str = ".") -> str:
        """
        Get path for next auto-save file.
        
        Args:
            base_dir: Directory for auto-save files
            
        Returns:
            Path to auto-save file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_dir}/autosave_{timestamp}.gsession"
    
    def cleanup_old_backups(self, base_dir: str = "."):
        """Remove old auto-save files beyond max_backups."""
        import os
        import glob
        
        pattern = f"{base_dir}/autosave_*.gsession"
        auto_saves = sorted(glob.glob(pattern))
        
        # Remove oldest files
        while len(auto_saves) > self.max_backups:
            oldest = auto_saves.pop(0)
            try:
                os.remove(oldest)
                print(f"Removed old auto-save: {oldest}")
            except:
                pass