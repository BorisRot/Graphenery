"""Configuration management for Graphenery application."""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class LatticeConfig:
    """Configuration for lattice generation."""
    bond_length: float = 1.0
    num_units: int = 7
    boundary: str = "bearded"
    x0_ns: float = 0.0
    x0_swne: float = -3.5
    x0_nwse: float = 3.5
    periodic_bc: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.bond_length <= 0:
            raise ValueError("Bond length must be positive")
        if self.num_units < 1:
            raise ValueError("Number of units must be at least 1")
        if self.boundary not in ["square", "bearded", "zigzag", "armchair", "mixed"]:
            raise ValueError(f"Invalid boundary condition: {self.boundary}")


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    node_size_factor: float = 50.0
    show_labels: bool = False
    edge_color: str = 'gray'
    a_sublattice_color: str = 'red'
    b_sublattice_color: str = 'black'
    
    def compute_node_size(self, num_units: int) -> float:
        """Compute node size based on number of units."""
        return self.node_size_factor * (7 / num_units) ** 1.5


@dataclass
class PhysicsConfig:
    """Configuration for physics calculations."""
    zm_epsilon: float = 1.0e-13
    bond_strength: float = 1.0
    disorder_type: str = "Gaussian"
    disorder_parameter: float = 0.0
    realization_num: int = 100
    
    def __post_init__(self):
        """Validate physics parameters."""
        if self.zm_epsilon <= 0:
            raise ValueError("ZM epsilon must be positive")
        if self.disorder_type not in ["Gaussian", "Uniform"]:
            raise ValueError(f"Invalid disorder type: {self.disorder_type}")


@dataclass
class PlotConfig:
    """Configuration for zero mode plotting."""
    plot_zm: bool = True
    plot_spectrum: bool = False
    plot_wavefunction: bool = False
    plot_interference: bool = False
    select_lines: bool = False
    make_fit: bool = False
    max_lines: int = 1
    resolution_factor: int = 1
    smoothing_factor: int = 1
    sublattice_factor: float = 1.0


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'lattice': asdict(self.lattice),
            'visualization': asdict(self.visualization),
            'physics': asdict(self.physics),
            'plot': asdict(self.plot)
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ApplicationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            lattice=LatticeConfig(**config_dict.get('lattice', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            physics=PhysicsConfig(**config_dict.get('physics', {})),
            plot=PlotConfig(**config_dict.get('plot', {}))
        )


class ChainConfig:
    """Configuration for different chain types."""
    
    CHAIN_COLORS = {
        'ns': {'A': 'orange', 'B': 'purple'},
        'swne': {'A': 'magenta', 'B': 'yellow'},
        'nwse': {'A': 'cyan', 'B': 'blue'}
    }
    
    @classmethod
    def get_colors(cls, chain_type: str) -> Dict[str, str]:
        """Get color configuration for chain type."""
        return cls.CHAIN_COLORS.get(chain_type, {'A': 'red', 'B': 'black'})