"""Polygon selection system for interactive lattice manipulation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Optional, Set
import json


class PolygonSelector:
    """Interactive polygon selection on matplotlib plots."""
    
    def __init__(self, ax: plt.Axes, line_color: str = 'red', 
                 line_width: float = 2.0, fill_alpha: float = 0.1):
        """
        Initialize polygon selector.
        
        Args:
            ax: Matplotlib axes to draw on
            line_color: Color of polygon outline
            line_width: Width of polygon line
            fill_alpha: Transparency of polygon fill
        """
        self.ax = ax
        self.line_color = line_color
        self.line_width = line_width
        self.fill_alpha = fill_alpha
        
        self.vertices = []  # List of (x, y) tuples
        self.is_complete = False
        self.is_active = False
        
        # Visual elements
        self.vertex_markers = []  # List of Line2D objects for vertices
        self.edge_lines = []      # List of Line2D objects for edges
        self.polygon_patch = None # Polygon patch when complete
        
        # Event connections
        self.cid_click = None
        self.cid_key = None
        self.cid_right_click = None
    
    def activate(self):
        """Activate polygon drawing mode."""
        self.is_active = True
        self.vertices = []
        self.is_complete = False
        self._clear_visuals()
        
        # Connect event handlers
        self.cid_click = self.ax.figure.canvas.mpl_connect(
            'button_press_event', self._on_click
        )
        self.cid_key = self.ax.figure.canvas.mpl_connect(
            'key_press_event', self._on_key
        )
        
        print("Polygon Selection Mode: Click to add vertices. Right-click or press ENTER to complete.")
    
    def deactivate(self):
        """Deactivate polygon drawing mode."""
        self.is_active = False
        self._disconnect_events()
    
    def _disconnect_events(self):
        """Disconnect all event handlers."""
        if self.cid_click:
            self.ax.figure.canvas.mpl_disconnect(self.cid_click)
        if self.cid_key:
            self.ax.figure.canvas.mpl_disconnect(self.cid_key)
        if self.cid_right_click:
            self.ax.figure.canvas.mpl_disconnect(self.cid_right_click)
    
    def _on_click(self, event):
        """Handle mouse click events."""
        if not self.is_active or self.is_complete:
            return
        
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        if event.button == 1:  # Left click - add vertex
            self._add_vertex(event.xdata, event.ydata)
        elif event.button == 3:  # Right click - complete polygon
            if len(self.vertices) >= 3:
                self.complete()
    
    def _on_key(self, event):
        """Handle keyboard events."""
        if not self.is_active or self.is_complete:
            return
        
        if event.key == 'enter':  # Complete polygon
            if len(self.vertices) >= 3:
                self.complete()
        elif event.key == 'escape':  # Cancel
            self.cancel()
    
    def _add_vertex(self, x: float, y: float):
        """Add a vertex to the polygon."""
        self.vertices.append((x, y))
        
        # Draw vertex marker
        marker = self.ax.plot(x, y, 'o', color=self.line_color, 
                             markersize=8, zorder=10)[0]
        self.vertex_markers.append(marker)
        
        # Draw edge to previous vertex
        if len(self.vertices) > 1:
            prev_x, prev_y = self.vertices[-2]
            line = self.ax.plot([prev_x, x], [prev_y, y], 
                               color=self.line_color, 
                               linewidth=self.line_width,
                               linestyle='--',
                               zorder=9)[0]
            self.edge_lines.append(line)
        
        self.ax.figure.canvas.draw_idle()
        print(f"Vertex added: ({x:.2f}, {y:.2f}). Total vertices: {len(self.vertices)}")
    
    def complete(self):
        """Complete the polygon."""
        if len(self.vertices) < 3:
            print("Need at least 3 vertices to complete polygon")
            return
        
        self.is_complete = True
        self.is_active = False
        self._disconnect_events()
        
        # Draw closing edge
        if len(self.vertices) > 2:
            first_x, first_y = self.vertices[0]
            last_x, last_y = self.vertices[-1]
            line = self.ax.plot([last_x, first_x], [last_y, first_y],
                               color=self.line_color,
                               linewidth=self.line_width,
                               linestyle='--',
                               zorder=9)[0]
            self.edge_lines.append(line)
        
        # Replace dashed lines with solid polygon
        self._draw_completed_polygon()
        
        self.ax.figure.canvas.draw_idle()
        print(f"Polygon completed with {len(self.vertices)} vertices")
    
    def _draw_completed_polygon(self):
        """Draw the completed polygon as a filled patch."""
        # Remove dashed edges
        for line in self.edge_lines:
            line.remove()
        self.edge_lines = []
        
        # Add polygon patch
        polygon_array = np.array(self.vertices)
        self.polygon_patch = MplPolygon(
            polygon_array,
            closed=True,
            edgecolor=self.line_color,
            facecolor=self.line_color,
            linewidth=self.line_width,
            alpha=self.fill_alpha,
            linestyle='-',
            zorder=8
        )
        self.ax.add_patch(self.polygon_patch)
    
    def cancel(self):
        """Cancel polygon drawing."""
        self._clear_visuals()
        self.vertices = []
        self.is_complete = False
        self.is_active = False
        self._disconnect_events()
        self.ax.figure.canvas.draw_idle()
        print("Polygon selection cancelled")
    
    def clear(self):
        """Clear the polygon from display."""
        self._clear_visuals()
        self.vertices = []
        self.is_complete = False
        self.ax.figure.canvas.draw_idle()
    
    def _clear_visuals(self):
        """Remove all visual elements."""
        for marker in self.vertex_markers:
            marker.remove()
        self.vertex_markers = []
        
        for line in self.edge_lines:
            line.remove()
        self.edge_lines = []
        
        if self.polygon_patch:
            self.polygon_patch.remove()
            self.polygon_patch = None
    
    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """
        Check which points are inside the polygon.
        
        Args:
            points: Nx2 array of (x, y) coordinates
            
        Returns:
            Boolean array of length N indicating which points are inside
        """
        if not self.is_complete or len(self.vertices) < 3:
            return np.zeros(len(points), dtype=bool)
        
        from matplotlib.path import Path
        polygon_path = Path(self.vertices)
        return polygon_path.contains_points(points)
    
    def get_interior_nodes(self, node_positions: Dict) -> Set[Tuple]:
        """
        Get nodes that are inside the polygon.
        
        Args:
            node_positions: Dictionary mapping node tuples to (x, y) positions
            
        Returns:
            Set of node tuples that are inside the polygon
        """
        if not self.is_complete:
            return set()
        
        nodes = list(node_positions.keys())
        positions = np.array([node_positions[node] for node in nodes])
        
        inside = self.contains_points(positions)
        return set(node for node, is_inside in zip(nodes, inside) if is_inside)
    
    def to_dict(self) -> Dict:
        """Serialize polygon to dictionary."""
        return {
            'vertices': self.vertices,
            'is_complete': self.is_complete,
            'line_color': self.line_color,
            'line_width': self.line_width,
            'fill_alpha': self.fill_alpha
        }
    
    @classmethod
    def from_dict(cls, data: Dict, ax: plt.Axes) -> 'PolygonSelector':
        """Deserialize polygon from dictionary."""
        selector = cls(
            ax,
            line_color=data.get('line_color', 'red'),
            line_width=data.get('line_width', 2.0),
            fill_alpha=data.get('fill_alpha', 0.1)
        )
        selector.vertices = data.get('vertices', [])
        selector.is_complete = data.get('is_complete', False)
        
        if selector.is_complete and len(selector.vertices) >= 3:
            # Reconstruct visual elements
            for x, y in selector.vertices:
                marker = ax.plot(x, y, 'o', color=selector.line_color,
                               markersize=8, zorder=10)[0]
                selector.vertex_markers.append(marker)
            
            selector._draw_completed_polygon()
        
        return selector


class OperationHistory:
    """Manages operation history for undo/redo functionality."""
    
    def __init__(self):
        """Initialize empty history."""
        self.operations = []  # List of operation dictionaries
        self.current_index = -1  # Index of current state (-1 means original)
    
    def add_operation(self, operation: Dict):
        """
        Add an operation to history.
        
        Args:
            operation: Dictionary describing the operation with keys:
                - 'type': Operation type ('extract', 'delete', 'highlight', etc.)
                - 'polygon': Polygon vertices
                - 'affected_nodes': Nodes affected by operation
                - 'timestamp': When operation was performed
                - Additional type-specific data
        """
        # If we're not at the end, remove forward history
        if self.current_index < len(self.operations) - 1:
            self.operations = self.operations[:self.current_index + 1]
        
        self.operations.append(operation)
        self.current_index += 1
    
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self.current_index >= 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self.current_index < len(self.operations) - 1
    
    def undo(self) -> Optional[Dict]:
        """
        Undo last operation.
        
        Returns:
            The operation that was undone, or None if can't undo
        """
        if not self.can_undo():
            return None
        
        operation = self.operations[self.current_index]
        self.current_index -= 1
        return operation
    
    def redo(self) -> Optional[Dict]:
        """
        Redo previously undone operation.
        
        Returns:
            The operation that was redone, or None if can't redo
        """
        if not self.can_redo():
            return None
        
        self.current_index += 1
        return self.operations[self.current_index]
    
    def get_current_state(self) -> List[Dict]:
        """Get list of operations up to current state."""
        return self.operations[:self.current_index + 1]
    
    def clear(self):
        """Clear all history."""
        self.operations = []
        self.current_index = -1
    
    def to_dict(self) -> Dict:
        """Serialize history to dictionary."""
        return {
            'operations': self.operations,
            'current_index': self.current_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OperationHistory':
        """Deserialize history from dictionary."""
        history = cls()
        history.operations = data.get('operations', [])
        history.current_index = data.get('current_index', -1)
        return history
    
    def save_to_file(self, filepath: str):
        """Save history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'OperationHistory':
        """Load history from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
