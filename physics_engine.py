"""Physics engine for quantum mechanical calculations on graphene lattices."""
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional
from functools import lru_cache
import hashlib
import pickle


class PhysicsEngine:
    """Handles quantum mechanical calculations for graphene lattices."""
    
    def __init__(self, zm_epsilon: float = 1.0e-10):
        """
        Initialize physics engine.
        
        Args:
            zm_epsilon: Threshold for identifying zero modes
        """
        self.zm_epsilon = zm_epsilon
        self._cache = {}
    
    def graph_to_hash(self, graph: nx.Graph) -> str:
        """Create a hash of graph structure for caching."""
        # Use canonical node ordering
        nodes = sorted(graph.nodes())
        edges = sorted(graph.edges())
        graph_str = f"{nodes}_{edges}"
        return hashlib.md5(graph_str.encode()).hexdigest()
    
    def compute_eigensystem(self, graph: nx.Graph, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors from graph adjacency matrix.
        
        Args:
            graph: NetworkX graph representing the lattice
            use_cache: Whether to use cached results
            
        Returns:
            Tuple of (adjacency_matrix, eigenvalues, eigenvectors)
        """
        graph_hash = self.graph_to_hash(graph) if use_cache else None
        
        if use_cache and graph_hash in self._cache:
            return self._cache[graph_hash]
        
        # Compute adjacency matrix
        #H = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        H = nx.to_numpy_array(graph, nodelist=graph.nodes())
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenmodes = np.linalg.eigh(H, UPLO='L')
        eigenmodes = eigenmodes.transpose()
        
        result = (H, eigenvals, eigenmodes)
        
        if use_cache and graph_hash:
            self._cache[graph_hash] = result
        
        return result
    
    def identify_zero_modes(self, eigenvals: np.ndarray, eigenmodes: np.ndarray, 
                           num_vacancies: int = 0) -> Dict[int, np.ndarray]:
        """
        Identify zero-energy modes and compute probability distributions.
        
        Args:
            eigenvals: Array of eigenvalues
            eigenmodes: Array of eigenvectors
            num_vacancies: Expected number of zero modes (sublattice imbalance)
            
        Returns:
            Dictionary mapping mode index to probability distribution
        """
        eigen_sorted = list(zip(eigenvals, eigenmodes))
        zm_vecs = {}
        zm_num = 1
        
        # Search around the middle of the spectrum
        # Clamp search window to valid index range
        n_eigs = len(eigenvals)
        mid_idx = n_eigs // 2
        
        # Search window should be wide enough but not exceed array bounds
        search_window = min(num_vacancies, mid_idx)
        
        neg_eigval_idx = max(0, mid_idx - search_window - 5)  # Add buffer
        pos_eigval_idx = min(n_eigs, mid_idx + search_window + 6)  # Add buffer
        
        for idx in range(neg_eigval_idx, pos_eigval_idx):
            if abs(eigenvals[idx]) < self.zm_epsilon:
                # Compute probability distribution (|ψ|²)
                zm_prob = eigen_sorted[idx][1] ** 2 * 5000
                
                # Filter small values
                zm_prob[zm_prob < self.zm_epsilon] = 0
                
                # Store wavefunction and probability
                zm_vecs[zm_num] = zm_prob
                zm_vecs[f'ZM_wf_{zm_num}'] = eigen_sorted[idx][1] * 5000
                zm_num += 1
        
        # Compute interference if multiple zero modes exist
        if zm_num > 2:  # More than one zero mode found
            self._compute_interference(zm_vecs, eigen_sorted, 
                                      neg_eigval_idx, pos_eigval_idx, eigenvals)
        
        return zm_vecs
        
        return zm_vecs
    
    def _compute_interference(self, zm_vecs: Dict, eigen_sorted: List, 
                            start_idx: int, end_idx: int, eigenvals: np.ndarray) -> None:
        """Compute interference patterns between zero modes."""
        zm_wavefunctions = []
        
        for idx in range(start_idx, end_idx):
            if abs(eigenvals[idx]) < self.zm_epsilon:
                zm_wavefunctions.append(eigen_sorted[idx][1])
        
        if len(zm_wavefunctions) > 1:
            # Constructive interference
            zm_interf = np.sum(zm_wavefunctions, axis=0) ** 2 * 5000
            zm_vecs['interf'] = zm_interf
            
            # Destructive interference (difference between first and last)
            zm_interf_minus = (zm_wavefunctions[-1] - zm_wavefunctions[0]) ** 2 * 5000
            zm_vecs['interfM'] = zm_interf_minus
    
    def disorder_average(self, graph: nx.Graph, disorder_type: str, 
                        disorder_param: float, bond_strength: float,
                        num_realizations: int, vacancies: List = None) -> Tuple[Dict, List]:
        """
        Perform disorder averaging over multiple realizations.
        
        Args:
            graph: Base graph structure
            disorder_type: Type of disorder ("Gaussian" or "Uniform")
            disorder_param: Disorder strength parameter
            bond_strength: Base bond strength
            num_realizations: Number of disorder realizations
            vacancies: List of vacancy sites
            
        Returns:
            Tuple of (averaged_zero_modes, averaged_eigenvalues)
        """
        if vacancies is None:
            vacancies = []
        
        edges_to_disorder = [(u, v, d['weight']) for u, v, d in graph.edges(data=True) 
                            if 'weight' in d]
        
        zm_prob_dict = {}
        eigenval_list = []
        
        print(f"Starting disorder averaging with {num_realizations} realizations...")
        
        for realization in range(num_realizations):
            # Create disordered graph
            disordered_graph = self._apply_disorder(
                graph, edges_to_disorder, disorder_type, 
                disorder_param, bond_strength
            )
            
            # Compute eigensystem
            H, eigenvals, eigenmodes = self.compute_eigensystem(disordered_graph, use_cache=False)
            eigenval_list.append(eigenvals)
            
            # Find zero modes
            zm_vecs = self.identify_zero_modes(eigenvals, eigenmodes, len(vacancies))
            zm_prob_dict[realization] = zm_vecs
            
            # Progress reporting
            if realization % max(1, num_realizations // 10) == 0:
                progress = 100 * realization / num_realizations
                print(f"Disorder averaging: {progress:.1f}% complete")
        
        # Average zero modes
        averaged_zm = self._average_zero_modes(zm_prob_dict, num_realizations)
        
        # Average eigenvalues
        averaged_eigenvals = np.mean(eigenval_list, axis=0).tolist()
        
        print("Disorder averaging complete!")
        return averaged_zm, averaged_eigenvals
    
    def _apply_disorder(self, graph: nx.Graph, edges: List, disorder_type: str,
                       disorder_param: float, bond_strength: float) -> nx.Graph:
        """Apply disorder to graph edges."""
        disordered_graph = nx.Graph()
        disordered_graph.add_nodes_from(graph.nodes(data=True))
        
        disordered_edges = []
        for u, v, weight in edges:
            disorder_value = self._get_disorder(disorder_type, disorder_param)
            new_weight = bond_strength + disorder_value
            disordered_edges.append((u, v, new_weight))
        
        disordered_graph.add_weighted_edges_from(disordered_edges)
        return disordered_graph
    
    @staticmethod
    def _get_disorder(disorder_type: str, disorder_param: float) -> float:
        """Generate disorder value based on type."""
        if disorder_type == "Gaussian":
            return np.random.normal(scale=disorder_param)
        elif disorder_type == "Uniform":
            return disorder_param * (np.random.random() - 0.5)
        return 0.0
    
    def _average_zero_modes(self, zm_prob_dict: Dict, num_realizations: int) -> Dict:
        """Average zero mode probabilities over realizations."""
        if not zm_prob_dict:
            return {}
        
        # Get structure from first realization
        first_realization = zm_prob_dict[0]
        averaged_zm = {}
        
        for key in first_realization.keys():
            if isinstance(key, int) or 'ZM_wf' in str(key):
                # Average this mode over all realizations
                mode_values = []
                for realization in zm_prob_dict.values():
                    if key in realization:
                        mode_values.append(realization[key])
                
                if mode_values:
                    averaged_zm[key] = np.mean(mode_values, axis=0)
        
        return averaged_zm
    
    def project_eigenvector(self, eigenvector: np.ndarray, line_start: Tuple, 
                          line_end: Tuple, node_positions: List, 
                          threshold: float = 1.1) -> List[Tuple]:
        """
        Project eigenvector onto a line.
        
        Args:
            eigenvector: Eigenvector to project
            line_start: Starting point of line
            line_end: Ending point of line
            node_positions: List of node positions
            threshold: Distance threshold for including nodes
            
        Returns:
            List of (distance, projection, value) tuples
        """
        line_vec = np.array(line_end) - np.array(line_start)
        line_vec /= np.linalg.norm(line_vec)
        
        eigenvector = np.array(eigenvector).flatten()
        projections = []
        
        for node_idx, vec_val in enumerate(eigenvector):
            if node_idx >= len(node_positions):
                continue
                
            node_pos = np.array(node_positions[node_idx])
            rel_pos = node_pos - np.array(line_start)
            
            # Perpendicular distance from node to line
            perpendicular_dist = np.linalg.norm(
                rel_pos - np.dot(rel_pos, line_vec) * line_vec
            )
            
            if perpendicular_dist > threshold:
                continue
            
            projection = np.dot(rel_pos, line_vec)
            distance = np.linalg.norm(rel_pos)
            projections.append((distance, projection, vec_val))
        
        # Sort by distance along line
        projections.sort(key=lambda x: x[0])
        return projections
    
    def clear_cache(self) -> None:
        """Clear the computation cache."""
        self._cache.clear()