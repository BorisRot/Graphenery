"""Visualization module for graphene lattices and zero modes."""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


class LatticeVisualizer:
    """Handles visualization of lattice structures."""
    
    def __init__(self, figure=None, ax=None):
        """
        Initialize visualizer.
        
        Args:
            figure: Matplotlib figure (created if None)
            ax: Matplotlib axes (created if None)
        """
        if figure is None:
            self.figure = plt.figure(figsize=(8, 8))
            self.ax = self.figure.add_subplot(111)
        else:
            self.figure = figure
            self.ax = ax if ax is not None else figure.add_subplot(111)
    
    def draw_lattice(self, graph: nx.Graph, positions: Dict, colors: Dict,
                    node_size: float = 50, edge_color: str = 'gray',
                    show_labels: bool = False, title: str = None) -> None:
        """
        Draw lattice structure.
        
        Args:
            graph: NetworkX graph
            positions: Node positions
            colors: Node colors
            node_size: Size of nodes
            edge_color: Color of edges
            show_labels: Whether to show node labels
            title: Plot title
        """
        self.ax.clear()
        
        color_list = [colors[node] for node in graph.nodes()]
        
        nx.draw(graph, positions,
               node_color=color_list,
               node_size=node_size,
               edge_color=edge_color,
               with_labels=show_labels,
               ax=self.ax)
        
        if title:
            self.ax.set_title(title)
        
        self.ax.axis('equal')
        self.figure.tight_layout()
    
    def draw_zero_mode(self, graph: nx.Graph, positions: Dict,
                      probabilities: np.ndarray, title: str = None) -> None:
        """
        Draw zero mode probability distribution.
        
        Args:
            graph: NetworkX graph
            positions: Node positions
            probabilities: Probability distribution
            title: Plot title
        """
        self.ax.clear()
        
        nx.draw(graph, positions,
               node_size=probabilities.tolist(),
               node_color='green',
               edge_color='gray',
               ax=self.ax)
        
        if title:
            self.ax.set_title(title)
        
        self.ax.axis('equal')
        self.figure.tight_layout()


class SpectrumVisualizer:
    """Visualize eigenvalue spectra."""
    
    @staticmethod
    def plot_spectrum(eigenvalues: List[float], title: str = "Energy Spectrum") -> plt.Figure:
        """
        Plot eigenvalue spectrum.
        
        Args:
            eigenvalues: List of eigenvalues
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        indices = np.arange(len(eigenvalues))
        ax.scatter(indices, eigenvalues, marker='o', s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Index', fontsize=14)
        ax.set_ylabel('Eigenvalue', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig


class WavefunctionVisualizer:
    """Visualize wavefunctions with interpolation and smoothing."""
    
    @staticmethod
    def plot_wavefunction(wavefunction: np.ndarray, positions: List[Tuple],
                         resolution_factor: int = 1, smoothing_factor: int = 1,
                         sublattice_factor: float = 1.0,
                         title: str = "Wavefunction") -> Tuple[plt.Figure, plt.Axes, plt.cm.ScalarMappable]:
        """
        Plot smoothed wavefunction.
        
        Args:
            wavefunction: Wavefunction values at nodes
            positions: Node positions
            resolution_factor: Resolution for interpolation
            smoothing_factor: Gaussian smoothing parameter
            sublattice_factor: Sublattice scaling factor
            title: Plot title
            
        Returns:
            Tuple of (figure, axes, image)
        """
        positions_array = np.array(positions)
        
        # Grid resolution
        a = 1.0 * sublattice_factor
        delta = a / resolution_factor
        
        # Create fine grid
        x_min, x_max = positions_array[:, 0].min() - delta, positions_array[:, 0].max() + delta
        y_min, y_max = positions_array[:, 1].min() - delta, positions_array[:, 1].max() + delta
        grid_x, grid_y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        
        # Interpolate
        wf_interpolated = griddata(
            positions_array, wavefunction, (grid_x, grid_y),
            method='cubic', fill_value=0
        )
        
        # Replace NaN with 0
        wf_interpolated = np.nan_to_num(wf_interpolated, nan=0.0)
        
        # Apply Gaussian smoothing
        if smoothing_factor > 0:
            sigma = sublattice_factor * (smoothing_factor / resolution_factor)
            wf_interpolated = gaussian_filter(wf_interpolated, sigma=sigma, mode="reflect")
        
        # Create plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(wf_interpolated,
                      extent=[y_min, y_max, x_min, x_max],
                      origin='lower',
                      aspect='auto',
                      cmap='viridis')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Wavefunction Intensity', fontsize=12)
        
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title(title, fontsize=16)
        
        fig.tight_layout()
        return fig, ax, im


class ProjectionVisualizer:
    """Visualize eigenvector projections along lines."""
    
    @staticmethod
    def plot_projection(projections: List[Tuple], title: str = "Projection",
                       make_fit: bool = False) -> plt.Figure:
        """
        Plot projection of wavefunction along line.
        
        Args:
            projections: List of (distance, projection, value) tuples
            title: Plot title
            make_fit: Whether to fit exponential and power law
            
        Returns:
            Matplotlib figure
        """
        if not projections:
            return None
        
        proj_x = [p[1] for p in projections]
        proj_y = [p[2] for p in projections]
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(proj_x, proj_y, 'b-', alpha=0.7, linewidth=2, label='Zero Mode Projection')
        
        if make_fit and len(projections) > 5:
            ProjectionVisualizer._add_fits(ax, proj_x, proj_y)
        
        ax.set_xlabel('Distance Along Line', fontsize=14)
        ax.set_ylabel('$|\\Psi|^2$', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _add_fits(ax: plt.Axes, x_data: List, y_data: List) -> None:
        """Add exponential and power law fits to projection."""
        try:
            # Find maximum
            max_idx = np.argmax(y_data)
            max_x = x_data[max_idx]
            max_y = y_data[max_idx]
            
            # Find peaks
            peaks_idx, _ = find_peaks(y_data, height=0, distance=1)
            
            if len(peaks_idx) < 2:
                return
            
            # Split data at maximum
            x_before = [x_data[i] for i in peaks_idx if i <= max_idx]
            y_before = [y_data[i] for i in peaks_idx if i <= max_idx]
            
            x_after = [x_data[i] for i in peaks_idx if i >= max_idx]
            y_after = [y_data[i] for i in peaks_idx if i >= max_idx]
            
            # Exponential fit (before maximum)
            if len(x_before) >= 2:
                def exp_model(x, a):
                    return max_y * np.exp(a * (x - max_x))
                
                params_exp, _ = curve_fit(exp_model, x_before, y_before, p0=[-1])
                
                x_fit = np.linspace(min(x_before), max(x_before), 100)
                y_fit = exp_model(x_fit, *params_exp)
                
                ax.plot(x_fit, y_fit, 'g--', linewidth=2,
                       label=f'Exp fit: ${max_y:.2f} \\exp({params_exp[0]:.2f}(x-{max_x:.2f}))$')
            
            # Power law fit (after maximum)
            if len(x_after) >= 2 and len(x_after) > 1:
                max_y_plus1 = y_after[1] if len(y_after) > 1 else y_after[0]
                
                def power_model(x, b):
                    return max_y_plus1 * np.power((x - max_x), b)
                
                params_pow, _ = curve_fit(power_model, x_after[1:], y_after[1:], p0=[1])
                
                x_fit = np.linspace(x_after[1], max(x_after), 100)
                y_fit = power_model(x_fit, *params_pow)
                
                ax.plot(x_fit, y_fit, 'r--', linewidth=2,
                       label=f'Power fit: ${max_y_plus1:.2f}(x-{max_x:.2f})^{{{params_pow[0]:.2f}}}$')
            
            # Mark peaks
            peak_x = [x_data[i] for i in peaks_idx]
            peak_y = [y_data[i] for i in peaks_idx]
            ax.scatter(peak_x, peak_y, color='red', s=50, zorder=5)
            
        except Exception as e:
            print(f"Fitting failed: {e}")


class InteractiveLineSelector:
    """Interactive line selection on plots."""
    
    def __init__(self, ax: plt.Axes, max_lines: int = 1):
        """
        Initialize line selector.
        
        Args:
            ax: Matplotlib axes
            max_lines: Maximum number of lines to draw
        """
        self.ax = ax
        self.max_lines = max_lines
        self.lines = []
        self.current_line = None
        self.selection_done = False
        
        self.cid_press = None
        self.cid_motion = None
        self.cid_release = None
    
    def select_lines(self) -> List[Tuple[Tuple, Tuple]]:
        """
        Allow user to draw lines on plot.
        
        Returns:
            List of (start_point, end_point) tuples
        """
        self.lines = []
        self.selection_done = False
        
        # Connect event handlers
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self._on_release)
        
        # Wait for selection
        while not self.selection_done:
            plt.pause(0.05)
        
        return self.lines
    
    def _on_press(self, event):
        """Handle mouse press event."""
        if event.inaxes != self.ax or len(self.lines) >= self.max_lines:
            return
        
        start_point = (event.xdata, event.ydata)
        line, = self.ax.plot([event.xdata], [event.ydata], 'b-', linewidth=2)
        self.current_line = {'start': start_point, 'line': line}
    
    def _on_motion(self, event):
        """Handle mouse motion event."""
        if event.inaxes != self.ax or self.current_line is None:
            return
        
        x0, y0 = self.current_line['start']
        self.current_line['line'].set_data([x0, event.xdata], [y0, event.ydata])
        self.ax.figure.canvas.draw()
    
    def _on_release(self, event):
        """Handle mouse release event."""
        if event.inaxes != self.ax or self.current_line is None:
            return
        
        end_point = (event.xdata, event.ydata)
        self.lines.append((self.current_line['start'], end_point))
        
        print(f"Line {len(self.lines)}: {self.current_line['start']} -> {end_point}")
        
        self.current_line = None
        
        if len(self.lines) >= self.max_lines:
            print(f"Maximum of {self.max_lines} lines reached.")
            self._disconnect()
            self.selection_done = True
    
    def _disconnect(self):
        """Disconnect event handlers."""
        if self.cid_press:
            self.ax.figure.canvas.mpl_disconnect(self.cid_press)
        if self.cid_motion:
            self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        if self.cid_release:
            self.ax.figure.canvas.mpl_disconnect(self.cid_release)


class ColorbarController:
    """Interactive colorbar control for wavefunction plots."""
    
    def __init__(self, figure: plt.Figure, image, colorbar):
        """
        Initialize colorbar controller.
        
        Args:
            figure: Matplotlib figure
            image: Image object
            colorbar: Colorbar object
        """
        self.figure = figure
        self.image = image
        self.colorbar = colorbar
        
        # Get initial limits
        self.vmin, self.vmax = image.get_clim()
        
        # Connect scroll event
        self.cid = figure.canvas.mpl_connect('scroll_event', self._on_scroll)
    
    def _on_scroll(self, event):
        """Handle mouse scroll for color scale adjustment."""
        if event.button == 'up':
            # Increase contrast
            self.vmin *= 0.9
            self.vmax *= 0.9
        elif event.button == 'down':
            # Decrease contrast
            self.vmin *= 1.1
            self.vmax *= 1.1
        
        self.image.set_clim(self.vmin, self.vmax)
        self.colorbar.update_normal(self.image)
        self.figure.canvas.draw_idle()
    
    def disconnect(self):
        """Disconnect scroll event."""
        if self.cid:
            self.figure.canvas.mpl_disconnect(self.cid)