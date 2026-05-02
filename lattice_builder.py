"""Fixed lattice construction module for graphene structures."""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


class LatticeBuilder(ABC):
    """Abstract base class for lattice builders."""
    
    def __init__(self, num_units: int, bond_length: float, bond_strength: float = 1.0):
        """Initialize lattice builder."""
        self.num_units = num_units
        self.bond_length = bond_length
        self.bond_strength = bond_strength
        self.a1 = np.array([-np.sqrt(3) / 2 * bond_length, 3 / 2 * bond_length])
        self.a2 = np.array([np.sqrt(3) / 2 * bond_length, 3 / 2 * bond_length])
        self.basis = np.array([[0, 0], [0, -bond_length]])
    
    @abstractmethod
    def build(self) -> Tuple[nx.Graph, Dict, Dict]:
        """Build the lattice structure."""
        pass
    
    def _create_base_lattice(self, boundary: str, periodic: bool = False):
        """Create base hexagonal lattice with specified boundary."""
        points_to_idx, boundaries = self._generate_points(boundary)
        graph, vertex_colors = self._generate_graph(points_to_idx, boundaries, periodic)
        positions = nx.get_node_attributes(graph, 'pos')
        return points_to_idx, boundaries, graph, vertex_colors, positions
    
    def _generate_points(self, boundary: str) -> Tuple[Dict, Dict]:
        """Generate lattice points within boundary."""
        n = self.num_units
        a = self.bond_length
        points_to_idx = {}
        boundaries = self._init_boundaries()
        idx = 0
        boundary_eps = 0.8 * a
        
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                R = i * self.a1 + j * self.a2
                pointA = tuple(R)
                pointB = tuple(R + self.basis[1])
                
                # Calculate polygon regions for both points
                polygonA = [
                    abs(pointA[1] + 1 / 2), 
                    abs(pointA[1] + 1 / 2 - (pointA[0] + np.sqrt(3) / 2) * np.sqrt(3)) / 2,
                    abs((pointA[1] + 1 / 2) + (pointA[0] + np.sqrt(3) / 2) * np.sqrt(3)) / 2
                ]
                polygonB = [
                    abs(pointB[1] + 1 / 2), 
                    abs(pointB[1] + 1 / 2 - (pointB[0] + np.sqrt(3) / 2) * np.sqrt(3)) / 2,
                    abs((pointB[1] + 1 / 2) + (pointB[0] + np.sqrt(3) / 2) * np.sqrt(3)) / 2
                ]
                
                regionA = np.max(polygonA)
                regionA_boundary = np.argmax(polygonA)
                regionB = np.max(polygonB)
                regionB_boundary = np.argmax(polygonB)
                
                # Check A sublattice point - pass sublattice info
                in_A, boundary_A = self._check_boundary(
                    pointA, polygonA, polygonB, boundary, n, a, i, j, sublattice='A'
                )

                # Check B sublattice point - pass sublattice info
                in_B, boundary_B = self._check_boundary(
                    pointB, polygonA, polygonB, boundary, n, a, i, j, sublattice='B'
                )

                if in_A:
                    points_to_idx[(i, j, 0)] = idx
                    idx += 1
                    self._register_boundary(boundaries, boundary, (i, j, 0),
                                          pointA, polygonA, polygonB, regionA, regionA_boundary,
                                          boundary_A, boundary_eps, 'A', i, j, n, a)

                if in_B:
                    points_to_idx[(i, j, 1)] = idx
                    idx += 1
                    self._register_boundary(boundaries, boundary, (i, j, 1),
                                          pointB, polygonA, polygonB, regionB, regionB_boundary,
                                          boundary_B, boundary_eps, 'B', i, j, n, a)

        return points_to_idx, boundaries

    def _check_boundary(self, point: Tuple, polygonA: List, polygonB: List,
                       boundary: str, n: int, a: float, i: int, j: int,
                       sublattice: str) -> Tuple[bool, int]:
        """
        Check if point is within specified boundary.

        Args:
            point: (x, y) coordinates of the point
            polygonA: Polygon coordinates for A sublattice point
            polygonB: Polygon coordinates for B sublattice point
            boundary: Boundary type
            n: Number of units
            a: Bond length
            i, j: Lattice indices
            sublattice: 'A' or 'B' to indicate which sublattice this point belongs to
        """
        x, y = point

        if boundary == "square":
            region = max(abs(x), abs(y))
            boundary_dir = 0 if abs(y) > abs(x) else 1
            return region <= (n - np.sqrt(3)/2 + 0.05) * a, boundary_dir

        elif boundary == "bearded":
            polygon = [
                abs(y + 0.5),
                abs(y + 0.5 - (x + np.sqrt(3)/2) * np.sqrt(3)) / 2,
                abs(y + 0.5 + (x + np.sqrt(3)/2) * np.sqrt(3)) / 2
            ]
            region = max(polygon)
            boundary_dir = polygon.index(region)
            return region <= (n - 0.5) * 1.5 * a, boundary_dir

        elif boundary == "zigzag":
            polygon = [
                abs(y + 0.5),
                abs(y + 0.5 - (x + np.sqrt(3)/2) * np.sqrt(3)) / 2,
                abs(y + 0.5 + (x + np.sqrt(3)/2) * np.sqrt(3)) / 2
            ]
            region = max(polygon)
            boundary_dir = polygon.index(region)
            return region <= (3/2) * n * a, boundary_dir

        elif boundary == "armchair":
            # CRITICAL FIX: Check sublattice-specific conditions
            regionA = np.max(polygonA)
            regionB = np.max(polygonB)
            polygon = [
                abs(y + 1/2),
                abs(y + 1/2 - (x + np.sqrt(3)/2) * np.sqrt(3)) / 2,
                abs(y + 1/2 + (x + np.sqrt(3)/2) * np.sqrt(3)) / 2
            ]
            region = max(polygon)
            boundary_dir = polygon.index(region)

            # Calculate all four arguments
            if n % 2 != 0:
                # Odd n case
                Aarg1 = (3/2) * n * a if abs(i) % 2 == 0 else (3/2) * (n - 1/2) * a
                Barg1 = (3/2) * (n - 1/2) * a if abs(j) % 2 != 0 else (3/2) * n * a
                Aarg2 = (3/2) * n * a if abs(i - j) % 2 == 0 else (3/2) * (n - 1/2) * a
                Barg2 = (3/2) * n * a if abs(i - j) % 2 == 0 else (3/2) * (n - 1/2) * a
            else:
                # Even n case
                Aarg1 = (3/2) * (n - 1/2) * a if (abs(j) % 2 != 0) and (abs(i) % 2 == 0) else (3/2) * n * a
                Barg1 = (3/2) * (n - 1/2) * a if (abs(j) % 2 == 0) and (abs(i) % 2 == 0) else (3/2) * n * a
                Aarg2 = (3/2) * (n - 1/2) * a if abs(i + j) % 2 == 0 else (3/2) * n * a
                Barg2 = (3/2) * (n - 1/2) * a if abs(i - j) % 2 == 0 else (3/2) * n * a

            # CRITICAL: Check based on which sublattice the point belongs to
            if sublattice == 'A':
                # For A sublattice: check regionA against Aarg1 and Aarg2
                in_boundary = (regionA <= Aarg1) or (regionA <= Aarg2)
            else:  # sublattice == 'B'
                # For B sublattice: check regionB against Barg1 and Barg2
                in_boundary = (regionB <= Barg1) or (regionB <= Barg2)

            return in_boundary, boundary_dir

        elif boundary == "mixed":
            polygon = [
                abs(y + 0.5),
                abs(y + 0.5 - (x + np.sqrt(3)/2) * np.sqrt(3)) / 2,
                abs(y + 0.5 + (x + np.sqrt(3)/2) * np.sqrt(3)) / 2
            ]
            region = max(polygon)
            boundary_dir = polygon.index(region)
            Aarg = (3/2) * n * a if abs(i - j) % 2 == 0 else (3/2) * (n - 0.5) * a
            return region <= Aarg, boundary_dir

        return False, 0

    def _init_boundaries(self) -> Dict:
        """Initialize boundary dictionaries."""
        boundaries = {}
        for btype in ['square', 'bearded', 'zigzag', 'armchair']:
            boundaries[f'{btype}A'] = {}
            boundaries[f'{btype}B'] = {}
            for direction in ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']:
                boundaries[f'{btype}A'][direction] = []
                boundaries[f'{btype}B'][direction] = []
        return boundaries

    def _register_boundary(self, boundaries: Dict, boundary: str, node: Tuple,
                          point: Tuple, polygonA: List, polygonB: List,
                          region: float, region_boundary: int,
                          boundary_dir: int, eps: float, sublattice: str,
                          i: int, j: int, n: int, a: float) -> None:
        """Register boundary nodes for periodic boundary conditions."""
        x, y = point

        if boundary == "square":
            is_boundary = abs(region - n * a) < eps

            if is_boundary:
                key = f'square{sublattice}'
                if region_boundary == 0 and y > 0:
                    boundaries[key]['N'].append(node)
                elif region_boundary == 0 and y < 0:
                    boundaries[key]['S'].append(node)
                elif region_boundary == 1 and x > 0:
                    boundaries[key]['E'].append(node)
                elif region_boundary == 1 and x < 0:
                    boundaries[key]['W'].append(node)

        elif boundary == "bearded":
            is_boundary = abs(region - (n - 0.5) * 1.5 * a) < eps

            if is_boundary:
                key = f'bearded{sublattice}'
                if boundary_dir == 0 and y > 0:
                    boundaries[key]['N'].append(node)
                elif boundary_dir == 0 and y < 0:
                    boundaries[key]['S'].append(node)
                elif boundary_dir == 1 and x > 0:
                    boundaries[key]['SW'].append(node)
                elif boundary_dir == 1 and x < 0:
                    boundaries[key]['NE'].append(node)
                elif boundary_dir == 2 and x > 0:
                    boundaries[key]['NW'].append(node)
                elif boundary_dir == 2 and x < 0:
                    boundaries[key]['SE'].append(node)

        elif boundary == "zigzag":
            is_boundary = abs(region - (3/2) * n * a) < eps

            if is_boundary:
                key = f'zigzag{sublattice}'
                if boundary_dir == 0 and y > 0:
                    boundaries[key]['N'].append(node)
                elif boundary_dir == 0 and y < 0:
                    boundaries[key]['S'].append(node)
                elif boundary_dir == 1 and x > 0:
                    boundaries[key]['SW'].append(node)
                elif boundary_dir == 1 and x < 0:
                    boundaries[key]['NE'].append(node)
                elif boundary_dir == 2 and x > 0:
                    boundaries[key]['NW'].append(node)
                elif boundary_dir == 2 and x < 0:
                    boundaries[key]['SE'].append(node)

        elif boundary == "armchair":
            # Calculate sublattice-specific arguments and check boundary
            regionA = np.max(polygonA)
            regionB = np.max(polygonB)

            if n % 2 != 0:
                Aarg1 = (3/2) * n * a if abs(i) % 2 == 0 else (3/2) * (n - 1/2) * a
                Barg1 = (3/2) * (n - 1/2) * a if abs(j) % 2 != 0 else (3/2) * n * a
                Aarg2 = (3/2) * n * a if abs(i - j) % 2 == 0 else (3/2) * (n - 1/2) * a
                Barg2 = (3/2) * n * a if abs(i - j) % 2 == 0 else (3/2) * (n - 1/2) * a
            else:
                Aarg1 = (3/2) * (n - 1/2) * a if (abs(j) % 2 != 0) and (abs(i) % 2 == 0) else (3/2) * n * a
                Barg1 = (3/2) * (n - 1/2) * a if (abs(j) % 2 == 0) and (abs(i) % 2 == 0) else (3/2) * n * a
                Aarg2 = (3/2) * (n - 1/2) * a if abs(i + j) % 2 == 0 else (3/2) * n * a
                Barg2 = (3/2) * (n - 1/2) * a if abs(i - j) % 2 == 0 else (3/2) * n * a

            # Check if on boundary based on sublattice
            if sublattice == 'A':
                is_boundary = (abs(regionA - Aarg1) < eps) or (abs(regionA - Aarg2) < eps)
            else:  # sublattice == 'B'
                is_boundary = (abs(regionB - Barg1) < eps) or (abs(regionB - Barg2) < eps)

            if is_boundary:
                key = f'armchair{sublattice}'
                if boundary_dir == 0 and y > 0:
                    boundaries[key]['N'].append(node)
                elif boundary_dir == 0 and y < 0:
                    boundaries[key]['S'].append(node)
                elif boundary_dir == 1 and x > 0:
                    boundaries[key]['SW'].append(node)
                elif boundary_dir == 1 and x < 0:
                    boundaries[key]['NE'].append(node)
                elif boundary_dir == 2 and x > 0:
                    boundaries[key]['NW'].append(node)
                elif boundary_dir == 2 and x < 0:
                    boundaries[key]['SE'].append(node)

    def _generate_graph(self, points_to_idx: Dict, boundaries: Dict, periodic: bool) -> Tuple[nx.Graph, Dict]:
        """Generate NetworkX graph from points."""
        G = nx.Graph()
        vertex_colors = {}

        for (i, j, sub), idx in points_to_idx.items():
            if sub == 0:
                pos = i * self.a1 + j * self.a2
                vertex_colors[(i, j, sub)] = 'red'
            else:
                pos = i * self.a1 + j * self.a2 + self.basis[1]
                vertex_colors[(i, j, sub)] = 'black'
            G.add_node((i, j, sub), pos=pos)

        for (i, j, sub) in points_to_idx.keys():
            if sub == 0:
                neighbors = [(i, j, 1), (i + 1, j, 1), (i, j + 1, 1)]
                for neighbor in neighbors:
                    if neighbor in points_to_idx:
                        G.add_edge((i, j, sub), neighbor, weight=self.bond_strength)

        if periodic:
            self._add_periodic_edges(G, boundaries)

        return G, vertex_colors

    def _add_periodic_edges(self, G: nx.Graph, boundaries: Dict) -> None:
        """Add periodic boundary condition edges."""
        tt = self.bond_strength
        boundary_keys = [key for key, value in boundaries.items() if any(value.values())]

        if not boundary_keys:
            return

        boundary_type = boundary_keys[0][:-1]

        if boundary_type == 'bearded':
            Abounds = boundaries.get('beardedA', {})
            Bbounds = boundaries.get('beardedB', {})

            if Abounds.get('N') and Bbounds.get('S'):
                pairs = list(zip(Abounds['N'], Bbounds['S'], [tt] * len(Bbounds['S'])))
                G.add_weighted_edges_from(pairs)

            if Bbounds.get('NW') and Abounds.get('SE'):
                pairs = list(zip(Bbounds['NW'], Abounds['SE'], [tt] * len(Abounds['SE'])))
                G.add_weighted_edges_from(pairs)

            if Bbounds.get('NE') and Abounds.get('SW'):
                pairs = list(zip(Bbounds['NE'], Abounds['SW'], [tt] * len(Abounds['SW'])))
                G.add_weighted_edges_from(pairs)

        elif boundary_type == 'zigzag':
            Abounds = boundaries.get('zigzagA', {})
            Bbounds = boundaries.get('zigzagB', {})

            if Bbounds.get('N') and Abounds.get('S'):
                pairs = list(zip(Bbounds['N'], Abounds['S'], [tt] * len(Abounds['S'])))
                G.add_weighted_edges_from(pairs)

            if Abounds.get('NW') and Bbounds.get('SE'):
                pairs = list(zip(Abounds['NW'], Bbounds['SE'], [tt] * len(Bbounds['SE'])))
                G.add_weighted_edges_from(pairs)

            if Abounds.get('NE') and Bbounds.get('SW'):
                pairs = list(zip(Abounds['NE'], Bbounds['SW'], [tt] * len(Bbounds['SW'])))
                G.add_weighted_edges_from(pairs)

        elif boundary_type == 'armchair':
            Abounds = boundaries.get('armchairA', {})
            Bbounds = boundaries.get('armchairB', {})

            if Bbounds.get('N') and Abounds.get('S'):
                pairs = list(zip(Bbounds['N'], Abounds['S'], [tt] * len(Abounds['S'])))
                G.add_weighted_edges_from(pairs)

            if Abounds.get('NW') and Bbounds.get('SE'):
                pairs = list(zip(Abounds['NW'], Bbounds['SE'], [tt] * len(Bbounds['SE'])))
                G.add_weighted_edges_from(pairs)

            if Abounds.get('NE') and Bbounds.get('SW'):
                pairs = list(zip(Abounds['NE'], Bbounds['SW'], [tt] * len(Bbounds['SW'])))
                G.add_weighted_edges_from(pairs)

        elif boundary_type == 'square':
            Abounds = boundaries.get('squareA', {})
            Bbounds = boundaries.get('squareB', {})

            if Abounds.get('N') and Bbounds.get('S'):
                pairs = list(zip(Abounds['N'], Bbounds['S'], [tt] * len(Bbounds['S'])))
                G.add_weighted_edges_from(pairs)

            if Abounds.get('W') and Bbounds.get('E'):
                pairs = list(zip(Abounds['W'], Bbounds['E'], [tt] * len(Bbounds['E'])))
                G.add_weighted_edges_from(pairs)

            if Bbounds.get('W') and Abounds.get('E'):
                pairs = list(zip(Bbounds['W'], Abounds['E'], [tt] * len(Abounds['E'])))
                G.add_weighted_edges_from(pairs)


class FullLatticeBuilder(LatticeBuilder):
    """Builder for complete graphene lattice."""

    def __init__(self, num_units: int, bond_length: float, boundary: str = "bearded",
                 periodic: bool = False, bond_strength: float = 1.0):
        super().__init__(num_units, bond_length, bond_strength)
        self.boundary = boundary
        self.periodic = periodic

    def build(self) -> Tuple[nx.Graph, Dict, Dict]:
        """Build full lattice."""
        _, _, graph, vertex_colors, positions = self._create_base_lattice(self.boundary, self.periodic)
        return graph, positions, vertex_colors


class ChainBuilder(LatticeBuilder):
    """Builder for chain-like structures."""

    def __init__(self, num_units: int, bond_length: float, chain_type: str,
                 x0: float = 0.0, bond_strength: float = 1.0):
        super().__init__(num_units, bond_length, bond_strength)
        self.chain_type = chain_type.lower()
        self.x0 = x0

    def build(self) -> Tuple[nx.Graph, Dict, Dict]:
        """Build chain structure."""
        points_to_idx, _, full_graph, full_colors, full_positions = self._create_base_lattice("bearded")

        # Get Aidx and Bidx from points_to_idx
        Aidx = [(i, j) for (i, j, sub) in points_to_idx.keys() if sub == 0]
        Bidx = [(i, j) for (i, j, sub) in points_to_idx.keys() if sub == 1]

        # Extract chain points using the proper method from ssh_gpt_functions.py
        chain_points = self._extract_chain_points_correct(
            Aidx, Bidx, points_to_idx, full_positions
        )

        chain_nodes = list(chain_points.keys())
        chain_graph = full_graph.subgraph(chain_nodes).copy()
        chain_positions = {node: full_positions[node] for node in chain_nodes}
        vertex_colors = self._assign_chain_colors(chain_graph)
        return chain_graph, chain_positions, vertex_colors

    def _extract_chain_points_correct(self, Aidx: List, Bidx: List,
                                     points_to_idx: Dict, positions: Dict) -> Dict:
        """
        Extract chain points using the exact algorithm from ssh_gpt_functions.py
        """
        N = self.num_units
        a = self.bond_length
        X0 = self.x0
        c1 = self.a1
        c2 = self.a2

        chain_points = {}

        if self.chain_type == 'ns':
            # NS chain - exact implementation from ssh_gpt_functions.py
            for mn in Aidx:
                pointA = tuple(mn[0] * c1 + mn[1] * c2)
                regionA = np.max([abs(pointA[1] + 1 / 2), abs(((3 / 2) * (N / 1.2)) * (pointA[0] + np.sqrt(3) / 2 - X0))])
                if regionA <= (3 / 2) * N * a:
                    chain_points[(mn[0], mn[1], 0)] = points_to_idx[(mn[0], mn[1], 0)]

            for mn in Bidx:
                pointB = tuple(mn[0] * c1 + mn[1] * c2)
                regionB = np.max([abs(pointB[1] - 1 / 2), abs(((3 / 2) * (N / 1.2)) * (pointB[0] + np.sqrt(3) / 2 - X0))])
                if regionB <= (3 / 2) * N * a:
                    chain_points[(mn[0], mn[1], 1)] = points_to_idx[(mn[0], mn[1], 1)]

        elif self.chain_type == 'swne':
            # SWNE chain - exact implementation from ssh_gpt_functions.py
            for mn in Aidx:
                pointA = tuple(mn[0] * c1 + mn[1] * c2)
                regionA = np.max([abs(pointA[1] + 1 / 2),
                    (3 / 2 * (N / 1.2)) * abs(pointA[1] + 1 / 2 - (1 / np.sqrt(3)) * (pointA[0] + np.sqrt(3) / 2 - X0))])
                if regionA <= (3 / 2) * N * a:
                    chain_points[(mn[0], mn[1], 0)] = points_to_idx[(mn[0], mn[1], 0)]

            for mn in Bidx:
                pointB = tuple(mn[0] * c1 + mn[1] * c2 + np.array([0, -1]))
                regionB = np.max([abs(pointB[1] + 1 / 2),
                    (3 / 2 * (N / 1.2)) * abs(pointB[1] + 1 / 2 - (1 / np.sqrt(3)) * (pointB[0] + np.sqrt(3) / 2 - X0))])
                if regionB <= (3 / 2) * N * a:
                    chain_points[(mn[0], mn[1], 1)] = points_to_idx[(mn[0], mn[1], 1)]

        elif self.chain_type == 'nwse':
            # NWSE chain - exact implementation from ssh_gpt_functions.py
            for mn in Aidx:
                pointA = tuple(mn[0] * c1 + mn[1] * c2)
                regionA = np.max([abs(pointA[1] + 1 / 2),
                    (3 / 2 * (N / 1.2)) * abs(pointA[1] + 1 / 2 + (1 / np.sqrt(3)) * (pointA[0] + np.sqrt(3) / 2 - X0))])
                if regionA <= (3 / 2) * N * a:
                    chain_points[(mn[0], mn[1], 0)] = points_to_idx[(mn[0], mn[1], 0)]

            for mn in Bidx:
                pointB = tuple(mn[0] * c1 + mn[1] * c2 + np.array([0, -1]))
                regionB = np.max([abs(pointB[1] + 1 / 2),
                    (3 / 2 * (N / 1.2)) * abs(pointB[1] + 1 / 2 + (1 / np.sqrt(3)) * (pointB[0] + np.sqrt(3) / 2 - X0))])
                if regionB <= (3 / 2) * N * a:
                    chain_points[(mn[0], mn[1], 1)] = points_to_idx[(mn[0], mn[1], 1)]

        return chain_points

    def _assign_chain_colors(self, graph: nx.Graph) -> Dict:
        """Assign colors to chain nodes."""
        color_map = {
            'ns': {'A': 'orange', 'B': 'purple'},
            'swne': {'A': 'magenta', 'B': 'yellow'},
            'nwse': {'A': 'cyan', 'B': 'blue'}
        }
        colors = color_map.get(self.chain_type, {'A': 'red', 'B': 'black'})
        vertex_colors = {}
        for node in graph.nodes():
            sublattice = 'A' if node[2] == 0 else 'B'
            vertex_colors[node] = colors[sublattice]
        return vertex_colors


class ChevronBuilder(LatticeBuilder):
    """Builder for chevron-like structures."""

    def __init__(self, num_units: int, bond_length: float,
                 orientation: str = 'ns', bond_strength: float = 1.0):
        """
        Initialize chevron builder.

        Args:
            num_units: Number of unit cells
            bond_length: Bond length (lattice constant)
            orientation: Chevron orientation - 'ns', 'swne', or 'nwse'
            bond_strength: Bond strength parameter
        """
        super().__init__(num_units, bond_length, bond_strength)
        self.orientation = orientation.lower()

        if self.orientation not in ['ns', 'swne', 'nwse']:
            raise ValueError(f"Invalid orientation: {orientation}. Must be 'ns', 'swne', or 'nwse'")

    def build(self) -> Tuple[nx.Graph, Dict, Dict]:
        """Build chevron structure by intersecting chains based on orientation."""

        # Default x0 positions for single chains
        x0_ns_default = 0.0
        x0_swne_default = -3.5
        x0_nwse_default = 3.5

        # Calculate spacing for parallel chains
        sqrt3 = np.sqrt(3)
        spacing_diagonal = sqrt3 * self.bond_length  # For SWNE and NWSE
        spacing_ns = 2.44  # For NS chains (approximately √3 * √2 or hexagon width)

        if self.orientation == 'ns':
            # NS Chevron: Two parallel NS chains intersecting with SWNE and NWSE
            ns_north = ChainBuilder(self.num_units, self.bond_length, 'ns',
                                   x0_ns_default - spacing_ns/2, self.bond_strength)
            ns_south = ChainBuilder(self.num_units, self.bond_length, 'ns',
                                   x0_ns_default + spacing_ns/2, self.bond_strength)
            swne = ChainBuilder(self.num_units, self.bond_length, 'swne',
                               x0_swne_default, self.bond_strength)
            nwse = ChainBuilder(self.num_units, self.bond_length, 'nwse',
                               x0_nwse_default, self.bond_strength)

            ns_north_graph, ns_north_pos, ns_north_colors = ns_north.build()
            ns_south_graph, ns_south_pos, ns_south_colors = ns_south.build()
            swne_graph, swne_pos, swne_colors = swne.build()
            nwse_graph, nwse_pos, nwse_colors = nwse.build()

            # Left arm: SWNE ∩ NS_north
            left_nodes = set(swne_graph.nodes()) & set(ns_north_graph.nodes())
            # Right arm: NWSE ∩ NS_south
            right_nodes = set(nwse_graph.nodes()) & set(ns_south_graph.nodes())

            # Use SWNE colors for left, NWSE colors for right
            arm_graphs = [(swne_graph, swne_pos, swne_colors),
                         (nwse_graph, nwse_pos, nwse_colors)]

        elif self.orientation == 'swne':
            # SWNE Chevron: Two parallel SWNE chains intersecting with NS and NWSE
            swne_north = ChainBuilder(self.num_units, self.bond_length, 'swne',
                                     x0_swne_default + spacing_diagonal, self.bond_strength)
            swne_south = ChainBuilder(self.num_units, self.bond_length, 'swne',
                                     x0_swne_default, self.bond_strength)
            ns = ChainBuilder(self.num_units, self.bond_length, 'ns',
                            x0_ns_default, self.bond_strength)
            nwse = ChainBuilder(self.num_units, self.bond_length, 'nwse',
                               x0_nwse_default, self.bond_strength)

            swne_north_graph, swne_north_pos, swne_north_colors = swne_north.build()
            swne_south_graph, swne_south_pos, swne_south_colors = swne_south.build()
            ns_graph, ns_pos, ns_colors = ns.build()
            nwse_graph, nwse_pos, nwse_colors = nwse.build()

            # Left arm: NS ∩ SWNE_north
            left_nodes = set(ns_graph.nodes()) & set(swne_north_graph.nodes())
            # Right arm: NWSE ∩ SWNE_south
            right_nodes = set(nwse_graph.nodes()) & set(swne_south_graph.nodes())

            # Use NS colors for left, NWSE colors for right
            arm_graphs = [(ns_graph, ns_pos, ns_colors),
                         (nwse_graph, nwse_pos, nwse_colors)]

        elif self.orientation == 'nwse':
            # NWSE Chevron: Two parallel NWSE chains intersecting with NS and SWNE
            nwse_north = ChainBuilder(self.num_units, self.bond_length, 'nwse',
                                     x0_nwse_default + spacing_diagonal, self.bond_strength)
            nwse_south = ChainBuilder(self.num_units, self.bond_length, 'nwse',
                                     x0_nwse_default, self.bond_strength)
            ns = ChainBuilder(self.num_units, self.bond_length, 'ns',
                            x0_ns_default, self.bond_strength)
            swne = ChainBuilder(self.num_units, self.bond_length, 'swne',
                               x0_swne_default, self.bond_strength)

            nwse_north_graph, nwse_north_pos, nwse_north_colors = nwse_north.build()
            nwse_south_graph, nwse_south_pos, nwse_south_colors = nwse_south.build()
            ns_graph, ns_pos, ns_colors = ns.build()
            swne_graph, swne_pos, swne_colors = swne.build()

            # Left arm: NS ∩ NWSE_north
            left_nodes = set(ns_graph.nodes()) & set(nwse_north_graph.nodes())
            # Right arm: SWNE ∩ NWSE_south
            right_nodes = set(swne_graph.nodes()) & set(nwse_south_graph.nodes())

            # Use NS colors for left, SWNE colors for right
            arm_graphs = [(ns_graph, ns_pos, ns_colors),
                         (swne_graph, swne_pos, swne_colors)]

        # Build the chevron graph
        chevron_graph = nx.Graph()
        chevron_pos = {}
        chevron_colors = {}

        # Add left arm
        for node in left_nodes:
            chevron_graph.add_node(node)
            chevron_pos[node] = arm_graphs[0][1][node]
            chevron_colors[node] = arm_graphs[0][2][node]

        # Add right arm
        for node in right_nodes:
            chevron_graph.add_node(node)
            chevron_pos[node] = arm_graphs[1][1][node]
            chevron_colors[node] = arm_graphs[1][2][node]

        # Add edges from left arm
        for u, v in arm_graphs[0][0].edges():
            if u in chevron_graph and v in chevron_graph:
                chevron_graph.add_edge(u, v, weight=self.bond_strength)

        # Add edges from right arm
        for u, v in arm_graphs[1][0].edges():
            if u in chevron_graph and v in chevron_graph:
                chevron_graph.add_edge(u, v, weight=self.bond_strength)

        return chevron_graph, chevron_pos, chevron_colors


class LatticeFactory:
    """Factory for creating different lattice types."""

    @staticmethod
    def create(lattice_type: str, num_units: int, bond_length: float,
               bond_strength: float = 1.0, **kwargs) -> Tuple[nx.Graph, Dict, Dict]:
        """Create lattice of specified type."""
        builders = {
            'full': FullLatticeBuilder,
            'ns': lambda nu, bl, bs: ChainBuilder(nu, bl, 'ns', kwargs.get('x0', 0.0), bs),
            'swne': lambda nu, bl, bs: ChainBuilder(nu, bl, 'swne', kwargs.get('x0', -3.5), bs),
            'nwse': lambda nu, bl, bs: ChainBuilder(nu, bl, 'nwse', kwargs.get('x0', 3.5), bs),
            'chevron': lambda nu, bl, bs: ChevronBuilder(nu, bl, kwargs.get('orientation', 'ns'), bs)
        }

        if lattice_type == 'full':
            builder = builders[lattice_type](num_units, bond_length, kwargs.get('boundary', 'bearded'),
                                            kwargs.get('periodic', False), bond_strength)
        elif lattice_type in ['ns', 'swne', 'nwse']:
            builder = builders[lattice_type](num_units, bond_length, bond_strength)
        elif lattice_type == 'chevron':
            builder = builders[lattice_type](num_units, bond_length, bond_strength)
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")

        return builder.build()


class LatticeModifier:
    """Modify existing lattice structures."""

    @staticmethod
    def merge_graphs(graph1: nx.Graph, graph2: nx.Graph, pos1: Dict, pos2: Dict,
                    colors1: Dict, colors2: Dict) -> Tuple[nx.Graph, Dict, Dict]:
        """Merge two graphs with their positions and colors."""
        merged_graph = nx.compose(graph1, graph2)
        merged_pos = {**pos1, **pos2}
        merged_colors = {**colors1, **colors2}
        nx.set_node_attributes(merged_graph, merged_pos, 'pos')
        return merged_graph, merged_pos, merged_colors

    @staticmethod
    def intersect_graphs(graphs: List[nx.Graph], positions: List[Dict],
                        colors: List[Dict]) -> Tuple[nx.Graph, Dict, Dict]:
        """Intersect multiple graphs."""
        if len(graphs) < 2:
            return graphs[0], positions[0], colors[0]

        inter_graph = nx.intersection(graphs[0], graphs[1])
        inter_pos = {x: positions[0][x] for x in positions[0] if x in positions[1]}
        inter_colors = {x: colors[0][x] for x in colors[0] if x in colors[1]}

        for i in range(2, len(graphs)):
            inter_graph = nx.intersection(inter_graph, graphs[i])
            inter_pos = {x: positions[i][x] for x in positions[i] if x in inter_pos}
            inter_colors = {x: colors[i][x] for x in colors[i] if x in inter_colors}

        nx.set_node_attributes(inter_graph, inter_pos, 'pos')
        return inter_graph, inter_pos, inter_colors

    @staticmethod
    def remove_node(graph: nx.Graph, positions: Dict, colors: Dict,
                   x: float, y: float, tolerance: float = 0.2) -> Optional[Tuple]:
        """Remove node near clicked position."""
        clicked_node = None
        for node, pos in positions.items():
            dist = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            if dist < tolerance:
                clicked_node = node
                break

        if clicked_node is not None:
            graph.remove_node(clicked_node)
            del positions[clicked_node]
            del colors[clicked_node]

        return clicked_node

    @staticmethod
    def add_unit_cell(graph: nx.Graph, positions: Dict, colors: Dict, x: float, y: float,
                     lattice_vectors: Tuple, bond_strength: float = 1.0, tolerance: float = 0.2) -> Optional[Tuple]:
        """Add unit cell near clicked position."""
        clicked_node = None
        for node, pos in positions.items():
            dist = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            if dist < tolerance:
                clicked_node = node
                break

        if clicked_node is None:
            return None

        a1, a2, basis = lattice_vectors
        d = np.array([0, -1])
        i, j, sub = clicked_node

        # Determine position relative to origin
        node_pos = positions[clicked_node]
        is_top = node_pos[1] > 0
        is_A = (sub == 0)

        # Add appropriate unit cell based on position and sublattice
        new_nodes = []

        if is_top and is_A:
            new_nodes = [
                (i + 1, j + 1, 0),
                (i + 1, j + 2, 1),
                (i + 2, j + 1, 1)
            ]
        elif is_top and not is_A:
            new_nodes = [
                (i + 1, j + 1, 1),
                (i, j + 1, 0),
                (i + 1, j, 0)
            ]
        elif not is_top and is_A:
            new_nodes = [
                (i - 1, j - 1, 0),
                (i, j - 1, 1),
                (i - 1, j, 1)
            ]
        elif not is_top and not is_A:
            new_nodes = [
                (i - 1, j - 1, 1),
                (i - 2, j - 1, 0),
                (i - 1, j - 2, 0)
            ]

        # Add nodes and edges
        for node in new_nodes:
            if node not in graph:
                node_i, node_j, node_sub = node
                if node_sub == 0:
                    pos = node_i * a1 + node_j * a2
                else:
                    pos = node_i * a1 + node_j * a2 + d

                graph.add_node(node)
                positions[node] = pos
                colors[node] = 'red' if node_sub == 0 else 'black'

        # Add edges between new nodes and existing structure
        for node in new_nodes:
            if node[2] == 0:  # A sublattice
                neighbors = [
                    (node[0], node[1], 1),
                    (node[0] + 1, node[1], 1),
                    (node[0], node[1] + 1, 1)
                ]
                for neighbor in neighbors:
                    if neighbor in graph:
                        graph.add_edge(node, neighbor, weight=bond_strength)

        return clicked_node