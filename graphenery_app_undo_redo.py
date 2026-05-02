"""Main GUI application for Graphenery."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

# Import our modules
from config import ApplicationConfig, LatticeConfig, VisualizationConfig, PhysicsConfig, PlotConfig
from lattice_builder import LatticeFactory, LatticeModifier
from physics_engine import PhysicsEngine
from visualizer import (LatticeVisualizer, SpectrumVisualizer, WavefunctionVisualizer,
                       ProjectionVisualizer, InteractiveLineSelector, ColorbarController)
from state_serializer import StateSerializer, WorkflowExporter, AutoSaveManager


class GrapheneryApp:
    """Main application class for Graphenery."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the application."""
        self.root = root
        self.root.title("Graphenery - Graphene Lattice Visualization")
        
        # Configuration
        self.config = ApplicationConfig()
        
        # Physics engine
        self.physics_engine = PhysicsEngine(self.config.physics.zm_epsilon)
        
        # Current state
        self.current_lattice = None
        self.current_graph = None
        self.current_positions = None
        self.current_colors = None
        self.vacancies = []
        
        # Original state (never modified - for undo to beginning)
        self.original_graph = None
        self.original_positions = None
        self.original_colors = None
        
        # Polygon selection system
        self.polygon_selectors = []  # List of polygon selectors (multiple polygons)
        self.active_polygon_selector = None  # Currently drawing polygon
        self.polygon_mode_active = False
        from polygon_selector import OperationHistory
        self.operation_history = OperationHistory()
        
        # Floating toolbar (created later)
        self.floating_toolbar = None
        
        # Setup UI
        self._setup_window()
        self._create_menu()
        self._create_ui()
        
        # Initial lattice
        self.generate_lattice()
    
    def _setup_window(self):
        """Setup window size and layout."""
        screen_height = self.root.winfo_screenheight()
        window_size = int(screen_height * 0.9)
        self.root.geometry(f"{window_size}x{window_size}")
    
    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Session management
        file_menu.add_command(label="Save Session", command=self.save_session, accelerator="Ctrl+S")
        file_menu.add_command(label="Save Session As...", command=self.save_session_as)
        file_menu.add_command(label="Load Session", command=self.load_session, accelerator="Ctrl+O")
        file_menu.add_separator()
        
        # Workflow export/import
        file_menu.add_command(label="Export Workflow", command=self.export_workflow)
        file_menu.add_command(label="Import Workflow", command=self.import_workflow)
        file_menu.add_separator()
        
        # Legacy config (keep for compatibility)
        file_menu.add_command(label="Save Configuration", command=self.save_config)
        file_menu.add_command(label="Load Configuration", command=self.load_config)
        file_menu.add_separator()
        
        file_menu.add_command(label="Export Image", command=self.export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-s>', lambda e: self.save_session())
        self.root.bind('<Control-o>', lambda e: self.load_session())
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo_operation, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo_operation, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Vacancies", command=self.clear_vacancies)
        
        # Bind undo/redo shortcuts
        self.root.bind('<Control-z>', lambda e: self.undo_operation())
        self.root.bind('<Control-y>', lambda e: self.redo_operation())
        self.root.bind('<Control-Shift-Z>', lambda e: self.redo_operation())  # Alternative
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Zero Mode Analysis", command=self.open_zero_mode_window)
        tools_menu.add_command(label="Clear Cache", command=self.clear_cache)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _create_ui(self):
        """Create main UI components."""
        # Get screen dimensions for proportional sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Left panel: 12% of screen width
        left_width = int(screen_width * 0.12)
        canvas_width = int(left_width * 0.85)  # Leave room for scrollbar
        
        # Left panel for controls with scrollbar
        left_container = ttk.Frame(self.root, width=left_width)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        left_container.pack_propagate(False)
        
        # Create canvas and scrollbar for scrollable left panel
        canvas = tk.Canvas(left_container, width=canvas_width)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        left_panel = ttk.Frame(canvas)
        
        left_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=left_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Right panel for plot
        right_panel = ttk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control sections
        self._create_lattice_controls(left_panel)
        self._create_options_controls(left_panel)
        self._create_action_buttons(left_panel)
        
        # Create plot area
        self._create_plot_area(right_panel)
        
        # Create floating toolbar (hidden initially)
        self._create_floating_toolbar()
    
    def _create_lattice_controls(self, parent):
        """Create lattice parameter controls."""
        frame = ttk.LabelFrame(parent, text="Lattice Parameters", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Bond length
        ttk.Label(frame, text="Bond Length:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bond_length_var = tk.DoubleVar(value=self.config.lattice.bond_length)
        ttk.Entry(frame, textvariable=self.bond_length_var, width=15).grid(row=0, column=1, pady=2)
        
        # Number of units
        ttk.Label(frame, text="Number of Units:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.num_units_var = tk.IntVar(value=self.config.lattice.num_units)
        ttk.Entry(frame, textvariable=self.num_units_var, width=15).grid(row=1, column=1, pady=2)
        
        # Boundary condition
        ttk.Label(frame, text="Boundary:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.boundary_var = tk.StringVar(value=self.config.lattice.boundary)
        boundary_combo = ttk.Combobox(frame, textvariable=self.boundary_var, 
                                     values=["square", "bearded", "zigzag", "armchair", "mixed"],
                                     width=13, state="readonly")
        boundary_combo.grid(row=2, column=1, pady=2)
        
        # Periodic BC
        self.periodic_var = tk.BooleanVar(value=self.config.lattice.periodic_bc)
        ttk.Checkbutton(frame, text="Periodic Boundary", 
                       variable=self.periodic_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # ZM Epsilon
        ttk.Label(frame, text="ZM Epsilon:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.zm_epsilon_var = tk.StringVar(value=f"{self.config.physics.zm_epsilon:.2e}")
        ttk.Entry(frame, textvariable=self.zm_epsilon_var, width=15).grid(row=4, column=1, pady=2)
    
    def _create_options_controls(self, parent):
        """Create lattice type selection controls."""
        frame = ttk.LabelFrame(parent, text="Lattice Type", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Lattice type selection
        self.lattice_type_var = tk.StringVar(value="full")
        
        types = [
            ("Full Lattice", "full"),
            ("NS Chain", "ns"),
            ("SWNE Chain", "swne"),
            ("NWSE Chain", "nwse"),
            ("NS + SWNE", "ns_swne"),
            ("NS + NWSE", "ns_nwse"),
            ("SWNE + NWSE", "swne_nwse"),
            ("Chevron", "chevron"),
            ("NS + Chevron", "ns_chevron"),
            ("SWNE + Chevron", "swne_chevron"),
            ("NWSE + Chevron", "nwse_chevron"),
            ("All Chains", "all_chains")
        ]
        
        for text, value in types:
            ttk.Radiobutton(frame, text=text, variable=self.lattice_type_var, 
                           value=value).pack(anchor=tk.W, pady=2)
        
        # Chain position sliders (for chain types) with discrete steps
        slider_frame = ttk.Frame(frame)
        slider_frame.pack(fill=tk.X, pady=5)
        
        # NS Chain Position with √3/2 steps
        ns_frame = ttk.Frame(slider_frame)
        ns_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ns_frame, text="NS Chain Position:").pack(side=tk.LEFT)
        self.x0_ns_label = ttk.Label(ns_frame, text="0.00")
        self.x0_ns_label.pack(side=tk.RIGHT)
        
        self.x0_ns_var = tk.DoubleVar(value=0.0)
        self.x0_ns_step = np.sqrt(3) / 2  # Discrete step for NS
        ns_scale = ttk.Scale(slider_frame, from_=-10, to=10, 
                            variable=self.x0_ns_var, orient=tk.HORIZONTAL,
                            command=lambda v: self._snap_slider('ns', float(v)))
        ns_scale.pack(fill=tk.X)
        
        # SWNE Chain Position with √3 steps
        swne_frame = ttk.Frame(slider_frame)
        swne_frame.pack(fill=tk.X, pady=2)
        ttk.Label(swne_frame, text="SWNE Chain Position:").pack(side=tk.LEFT)
        self.x0_swne_label = ttk.Label(swne_frame, text="-3.50")
        self.x0_swne_label.pack(side=tk.RIGHT)
        
        self.x0_swne_var = tk.DoubleVar(value=-3.5)
        self.x0_swne_step = np.sqrt(3)  # Discrete step for SWNE
        swne_scale = ttk.Scale(slider_frame, from_=-10, to=10, 
                              variable=self.x0_swne_var, orient=tk.HORIZONTAL,
                              command=lambda v: self._snap_slider('swne', float(v)))
        swne_scale.pack(fill=tk.X)
        
        # NWSE Chain Position with √3 steps
        nwse_frame = ttk.Frame(slider_frame)
        nwse_frame.pack(fill=tk.X, pady=2)
        ttk.Label(nwse_frame, text="NWSE Chain Position:").pack(side=tk.LEFT)
        self.x0_nwse_label = ttk.Label(nwse_frame, text="3.50")
        self.x0_nwse_label.pack(side=tk.RIGHT)
        
        self.x0_nwse_var = tk.DoubleVar(value=3.5)
        self.x0_nwse_step = np.sqrt(3)  # Discrete step for NWSE
        nwse_scale = ttk.Scale(slider_frame, from_=-10, to=10, 
                              variable=self.x0_nwse_var, orient=tk.HORIZONTAL,
                              command=lambda v: self._snap_slider('nwse', float(v)))
        nwse_scale.pack(fill=tk.X)
        
        # Chevron orientation selector
        chevron_frame = ttk.Frame(frame)
        chevron_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(chevron_frame, text="Chevron Orientation:").pack(anchor=tk.W)
        self.chevron_orientation_var = tk.StringVar(value="ns")
        chevron_combo = ttk.Combobox(chevron_frame, textvariable=self.chevron_orientation_var,
                                    values=["ns", "swne", "nwse"],
                                    width=13, state="readonly")
        chevron_combo.pack(anchor=tk.W, pady=2)
        
        # Special options
        options_frame = ttk.LabelFrame(parent, text="Special Options", padding=10)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.vacancy_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Vacancy Mode (click to remove)", 
                       variable=self.vacancy_mode_var).pack(anchor=tk.W, pady=2)
        
        self.add_unit_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Add Unit Mode (click to add)", 
                       variable=self.add_unit_mode_var).pack(anchor=tk.W, pady=2)
    
    def _snap_slider(self, chain_type: str, value: float):
        """Snap slider to discrete values and update label."""
        if chain_type == 'ns':
            step = self.x0_ns_step
            snapped = round(value / step) * step
            self.x0_ns_var.set(snapped)
            self.x0_ns_label.config(text=f"{snapped:.2f}")
        elif chain_type == 'swne':
            step = self.x0_swne_step
            snapped = round(value / step) * step
            self.x0_swne_var.set(snapped)
            self.x0_swne_label.config(text=f"{snapped:.2f}")
        elif chain_type == 'nwse':
            step = self.x0_nwse_step
            snapped = round(value / step) * step
            self.x0_nwse_var.set(snapped)
            self.x0_nwse_label.config(text=f"{snapped:.2f}")
    
    def _create_action_buttons(self, parent):
        """Create action buttons."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(frame, text="Generate Lattice", 
                  command=self.generate_lattice).pack(fill=tk.X, pady=2)
        
        # Polygon Selection Mode button
        ttk.Button(frame, text="Polygon Selection Mode", 
                  command=self.toggle_polygon_toolbar,
                  style='Accent.TButton').pack(fill=tk.X, pady=2)
        
        ttk.Button(frame, text="Zero Mode Analysis", 
                  command=self.open_zero_mode_window).pack(fill=tk.X, pady=2)
        
        ttk.Button(frame, text="Clear Vacancies", 
                  command=self.clear_vacancies).pack(fill=tk.X, pady=2)
    
    def _create_plot_area(self, parent):
        """Create matplotlib plot area."""
        self.figure = plt.Figure(figsize=(8, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()
        
        # Connect click events
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
    
    def _create_floating_toolbar(self):
        """Create draggable floating toolbar for polygon operations."""
        self.floating_toolbar = tk.Toplevel(self.root)
        self.floating_toolbar.title("Polygon Operations")
        self.floating_toolbar.geometry("280x550+100+100")
        
        # Make it stay on top but not modal
        self.floating_toolbar.attributes('-topmost', True)
        
        # Make draggable
        self.floating_toolbar.bind('<Button-1>', self._start_toolbar_drag)
        self.floating_toolbar.bind('<B1-Motion>', self._drag_toolbar)
        self._toolbar_drag_data = {"x": 0, "y": 0}
        
        # Container frame
        main_frame = ttk.Frame(self.floating_toolbar, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label (draggable handle)
        title_label = ttk.Label(main_frame, text="⋮⋮ Polygon Selection ⋮⋮", 
                               font=('Arial', 10, 'bold'))
        title_label.pack(pady=(0, 10))
        title_label.bind('<Button-1>', self._start_toolbar_drag)
        title_label.bind('<B1-Motion>', self._drag_toolbar)
        
        # Drawing Controls
        draw_frame = ttk.LabelFrame(main_frame, text="Drawing", padding=10)
        draw_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(draw_frame, text="Start Drawing Polygon", 
                  command=self.start_polygon_drawing).pack(fill=tk.X, pady=2)
        ttk.Button(draw_frame, text="Complete Polygon (Enter)",
                  command=self.complete_polygon).pack(fill=tk.X, pady=2)
        ttk.Button(draw_frame, text="Cancel (Esc)",
                  command=self.cancel_polygon).pack(fill=tk.X, pady=2)
        
        self.polygon_status_label = ttk.Label(draw_frame, text="Status: Inactive", 
                                             foreground="gray")
        self.polygon_status_label.pack(pady=2)
        
        # Style Controls
        style_frame = ttk.LabelFrame(main_frame, text="Polygon Style", padding=10)
        style_frame.pack(fill=tk.X, pady=5)
        
        # Color
        color_frame = ttk.Frame(style_frame)
        color_frame.pack(fill=tk.X, pady=2)
        ttk.Label(color_frame, text="Line Color:").pack(side=tk.LEFT)
        self.polygon_color_var = tk.StringVar(value="red")
        color_entry = ttk.Entry(color_frame, textvariable=self.polygon_color_var, width=10)
        color_entry.pack(side=tk.RIGHT)
        
        # Width
        width_frame = ttk.Frame(style_frame)
        width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(width_frame, text="Line Width:").pack(side=tk.LEFT)
        self.polygon_width_label = ttk.Label(width_frame, text="2.0")
        self.polygon_width_label.pack(side=tk.RIGHT)
        
        self.polygon_width_var = tk.DoubleVar(value=2.0)
        width_scale = ttk.Scale(style_frame, from_=0.5, to=5.0,
                               variable=self.polygon_width_var, orient=tk.HORIZONTAL,
                               command=lambda v: self.polygon_width_label.config(text=f"{float(v):.1f}"))
        width_scale.pack(fill=tk.X, pady=2)
        
        # Operations
        ops_frame = ttk.LabelFrame(main_frame, text="Operations", padding=10)
        ops_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(ops_frame, text="Extract Subgraph",
                  command=self.extract_subgraph).pack(fill=tk.X, pady=2)
        ttk.Button(ops_frame, text="Delete Interior Nodes",
                  command=self.delete_interior).pack(fill=tk.X, pady=2)
        
        # Highlight with color chooser
        highlight_frame = ttk.Frame(ops_frame)
        highlight_frame.pack(fill=tk.X, pady=2)
        ttk.Button(highlight_frame, text="Highlight Interior",
                  command=self.highlight_interior).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.highlight_color_var = tk.StringVar(value="yellow")
        ttk.Entry(highlight_frame, textvariable=self.highlight_color_var, 
                 width=8).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(ops_frame, text="Export Node List",
                  command=self.export_nodes).pack(fill=tk.X, pady=2)
        
        # History Controls
        history_frame = ttk.LabelFrame(main_frame, text="History", padding=10)
        history_frame.pack(fill=tk.X, pady=5)
        
        # Undo/Redo buttons in same row
        undo_redo_frame = ttk.Frame(history_frame)
        undo_redo_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(undo_redo_frame, text="Undo (Ctrl+Z)",
                  command=self.undo_operation).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(undo_redo_frame, text="Redo (Ctrl+Y)",
                  command=self.redo_operation).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        self.history_status_label = ttk.Label(history_frame, 
                                             text="Operations: 0", 
                                             foreground="blue")
        self.history_status_label.pack(pady=2)
        
        # Debug info section
        debug_frame = ttk.LabelFrame(main_frame, text="Debug Info", padding=10)
        debug_frame.pack(fill=tk.X, pady=5)
        
        self.debug_nodes_label = ttk.Label(debug_frame, text="Nodes: -", foreground="green")
        self.debug_nodes_label.pack(pady=1)
        
        self.debug_sublattice_label = ttk.Label(debug_frame, text="N_A: -, N_B: -", foreground="green")
        self.debug_sublattice_label.pack(pady=1)
        
        self.debug_imbalance_label = ttk.Label(debug_frame, text="Imbalance: -", foreground="green")
        self.debug_imbalance_label.pack(pady=1)
        
        # Cleanup
        cleanup_frame = ttk.Frame(main_frame)
        cleanup_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(cleanup_frame, text="Clear Polygon",
                  command=self.clear_polygon).pack(fill=tk.X, pady=2)
        ttk.Button(cleanup_frame, text="Close Toolbar",
                  command=self.floating_toolbar.withdraw).pack(fill=tk.X, pady=2)
        
        # Hide initially
        self.floating_toolbar.withdraw()
        
        # Handle window close
        self.floating_toolbar.protocol("WM_DELETE_WINDOW", self.floating_toolbar.withdraw)
    
    def _start_toolbar_drag(self, event):
        """Start dragging the toolbar."""
        self._toolbar_drag_data["x"] = event.x
        self._toolbar_drag_data["y"] = event.y
    
    def _drag_toolbar(self, event):
        """Drag the toolbar to new position."""
        x = self.floating_toolbar.winfo_x() - self._toolbar_drag_data["x"] + event.x
        y = self.floating_toolbar.winfo_y() - self._toolbar_drag_data["y"] + event.y
        self.floating_toolbar.geometry(f"+{x}+{y}")
    
    def generate_lattice(self):
        """Generate lattice based on current settings."""
        try:
            # Update config from UI
            self.config.lattice.bond_length = self.bond_length_var.get()
            self.config.lattice.num_units = self.num_units_var.get()
            self.config.lattice.boundary = self.boundary_var.get()
            self.config.lattice.periodic_bc = self.periodic_var.get()
            self.config.physics.zm_epsilon = float(self.zm_epsilon_var.get())
            
            # Update physics engine epsilon
            self.physics_engine.zm_epsilon = self.config.physics.zm_epsilon
            
            # Build lattice
            lattice_type = self.lattice_type_var.get()
            
            if lattice_type in ['full', 'ns', 'swne', 'nwse', 'chevron']:
                # Simple lattice types
                x0_map = {
                    'ns': self.x0_ns_var.get(),
                    'swne': self.x0_swne_var.get(),
                    'nwse': self.x0_nwse_var.get()
                }
                
                # For chevron, add orientation parameter
                if lattice_type == 'chevron':
                    self.current_graph, self.current_positions, self.current_colors = LatticeFactory.create(
                        lattice_type,
                        self.config.lattice.num_units,
                        self.config.lattice.bond_length,
                        self.config.physics.bond_strength,
                        orientation=self.chevron_orientation_var.get()
                    )
                else:
                    self.current_graph, self.current_positions, self.current_colors = LatticeFactory.create(
                        lattice_type,
                        self.config.lattice.num_units,
                        self.config.lattice.bond_length,
                        self.config.physics.bond_strength,
                        boundary=self.config.lattice.boundary,
                        periodic=self.config.lattice.periodic_bc,
                        x0=x0_map.get(lattice_type, 0.0)
                    )
            
            elif lattice_type == 'ns_swne':
                # NS + SWNE combination
                self.current_graph, self.current_positions, self.current_colors = \
                    self._build_two_chains('ns', 'swne')
            
            elif lattice_type == 'ns_nwse':
                # NS + NWSE combination
                self.current_graph, self.current_positions, self.current_colors = \
                    self._build_two_chains('ns', 'nwse')
            
            elif lattice_type == 'swne_nwse':
                # SWNE + NWSE combination
                self.current_graph, self.current_positions, self.current_colors = \
                    self._build_two_chains('swne', 'nwse')
            
            elif lattice_type == 'ns_chevron':
                # NS chain with chevron
                self.current_graph, self.current_positions, self.current_colors = \
                    self._build_chain_with_chevron('ns')
            
            elif lattice_type == 'swne_chevron':
                # SWNE chain with chevron
                self.current_graph, self.current_positions, self.current_colors = \
                    self._build_chain_with_chevron('swne')
            
            elif lattice_type == 'nwse_chevron':
                # NWSE chain with chevron
                self.current_graph, self.current_positions, self.current_colors = \
                    self._build_chain_with_chevron('nwse')
            
            elif lattice_type == 'all_chains':
                # All three chains merged
                self.current_graph, self.current_positions, self.current_colors = \
                    self._build_all_chains()
            
            else:
                raise ValueError(f"Unknown lattice type: {lattice_type}")
            
            # Clear vacancies and history when generating new lattice
            self.vacancies = []
            self.operation_history.clear()
            
            # Save as original state
            self._save_original_state()
            
            # Draw lattice
            self.draw_current_lattice()
            
            print(f"Generated {lattice_type} lattice with {len(self.current_graph.nodes())} nodes")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate lattice: {str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_original_state(self):
        """Save original graph state before any operations."""
        import networkx as nx
        self.original_graph = self.current_graph.copy()
        self.original_positions = self.current_positions.copy()
        self.original_colors = self.current_colors.copy()
        print("Original state saved")
    
    def _apply_operation_history(self):
        """Replay all operations from history to rebuild current state."""
        import networkx as nx
        
        # Start from original
        self.current_graph = self.original_graph.copy()
        self.current_positions = self.original_positions.copy()
        self.current_colors = self.original_colors.copy()
        
        # Apply each operation in order
        for operation in self.operation_history.get_current_state():
            self._apply_single_operation(operation)
        
        print(f"Applied {len(self.operation_history.get_current_state())} operations from history")
    
    def _apply_single_operation(self, operation: dict):
        """
        Apply a single operation to current graph.
        
        Args:
            operation: Dictionary with 'type' and operation-specific data
        """
        op_type = operation['type']
        
        if op_type == 'extract':
            # Keep only nodes in nodes_to_keep
            nodes_to_keep = set(operation['nodes_to_keep'])
            nodes_to_remove = set(self.current_graph.nodes()) - nodes_to_keep
            
            for node in nodes_to_remove:
                if node in self.current_graph:
                    self.current_graph.remove_node(node)
                if node in self.current_positions:
                    del self.current_positions[node]
                if node in self.current_colors:
                    del self.current_colors[node]
            
            print(f"Extract: Kept {len(nodes_to_keep)} nodes, removed {len(nodes_to_remove)} nodes")
        
        elif op_type == 'delete':
            # Remove nodes in nodes_to_delete
            nodes_to_delete = operation['nodes_to_delete']
            
            for node in nodes_to_delete:
                if node in self.current_graph:
                    self.current_graph.remove_node(node)
                if node in self.current_positions:
                    del self.current_positions[node]
                if node in self.current_colors:
                    del self.current_colors[node]
            
            print(f"Delete: Removed {len(nodes_to_delete)} nodes")
        
        elif op_type == 'highlight':
            # Change colors of nodes
            nodes_to_highlight = operation['nodes_to_highlight']
            highlight_color = operation['highlight_color']
            
            for node in nodes_to_highlight:
                if node in self.current_colors:
                    self.current_colors[node] = highlight_color
            
            print(f"Highlight: Changed color of {len(nodes_to_highlight)} nodes to {highlight_color}")
        
        elif op_type == 'vacancy':
            # Single node removal (compatibility with existing vacancy system)
            node = operation['node']
            if node in self.current_graph:
                self.current_graph.remove_node(node)
            if node in self.current_positions:
                del self.current_positions[node]
            if node in self.current_colors:
                del self.current_colors[node]
            
            print(f"Vacancy: Removed node {node}")
    
    # ========== Polygon Selection Methods ==========
    
    def toggle_polygon_toolbar(self):
        """Toggle visibility of floating polygon toolbar."""
        if self.floating_toolbar.winfo_viewable():
            self.floating_toolbar.withdraw()
        else:
            self.floating_toolbar.deiconify()
            # Position near the plot
            plot_x = self.root.winfo_x() + self.root.winfo_width() - 300
            plot_y = self.root.winfo_y() + 100
            self.floating_toolbar.geometry(f"+{plot_x}+{plot_y}")
    
    def start_polygon_drawing(self):
        """Start drawing a new polygon."""
        if self.current_graph is None:
            messagebox.showwarning("Warning", "Please generate a lattice first")
            return
        
        # Create new polygon selector
        from polygon_selector import PolygonSelector
        
        color = self.polygon_color_var.get()
        width = self.polygon_width_var.get()
        
        self.active_polygon_selector = PolygonSelector(self.ax, color, width)
        self.active_polygon_selector.activate()
        self.polygon_mode_active = True
        
        # Update status
        self.polygon_status_label.config(
            text=f"Status: Drawing polygon #{len(self.polygon_selectors) + 1}...", 
            foreground="green"
        )
        
        print(f"Started drawing polygon #{len(self.polygon_selectors) + 1}")
    
    def complete_polygon(self):
        """Complete the current polygon and add to list."""
        if self.active_polygon_selector and self.active_polygon_selector.is_active:
            self.active_polygon_selector.complete()
            self.polygon_mode_active = False
            
            # Add to list of polygons
            self.polygon_selectors.append(self.active_polygon_selector)
            
            # Update status
            vertices_count = len(self.active_polygon_selector.vertices)
            total_polygons = len(self.polygon_selectors)
            self.polygon_status_label.config(
                text=f"Status: {total_polygons} polygon(s) complete", 
                foreground="blue"
            )
            
            # Clear active reference
            self.active_polygon_selector = None
            
            self.canvas.draw()
            print(f"Polygon #{total_polygons} completed with {vertices_count} vertices")
    
    def cancel_polygon(self):
        """Cancel polygon drawing."""
        if self.active_polygon_selector:
            self.active_polygon_selector.cancel()
            self.active_polygon_selector = None
            self.polygon_mode_active = False
            
            # Update status
            total_polygons = len(self.polygon_selectors)
            self.polygon_status_label.config(
                text=f"Status: Cancelled ({total_polygons} polygon(s) remain)", 
                foreground="gray"
            )
            
            self.canvas.draw()
    
    def extract_subgraph(self):
        """Extract subgraph of nodes inside any polygon (union of all polygons)."""
        if not self._check_polygon_ready():
            return
        
        # Get nodes inside any polygon (union)
        interior_nodes = set()
        for polygon_selector in self.polygon_selectors:
            if polygon_selector.is_complete:
                nodes_in_polygon = polygon_selector.get_interior_nodes(self.current_positions)
                interior_nodes.update(nodes_in_polygon)
        
        if not interior_nodes:
            messagebox.showinfo("Info", "No nodes found inside polygons")
            return
        
        # Create operation
        import datetime
        operation = {
            'type': 'extract',
            'polygons': [ps.vertices.copy() for ps in self.polygon_selectors if ps.is_complete],
            'nodes_to_keep': list(interior_nodes),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add to history
        self.operation_history.add_operation(operation)
        
        # Apply operation
        self._apply_single_operation(operation)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Extracted subgraph: {len(interior_nodes)} nodes kept from {len(self.polygon_selectors)} polygon(s)")
        messagebox.showinfo("Success", f"Extracted subgraph with {len(interior_nodes)} nodes")
    
    def delete_interior(self):
        """Delete nodes inside any polygon (union of all polygons)."""
        if not self._check_polygon_ready():
            return
        
        # Get nodes inside any polygon (union)
        interior_nodes = set()
        for polygon_selector in self.polygon_selectors:
            if polygon_selector.is_complete:
                nodes_in_polygon = polygon_selector.get_interior_nodes(self.current_positions)
                interior_nodes.update(nodes_in_polygon)
        
        if not interior_nodes:
            messagebox.showinfo("Info", "No nodes found inside polygons")
            return
        
        # Create operation
        import datetime
        operation = {
            'type': 'delete',
            'polygons': [ps.vertices.copy() for ps in self.polygon_selectors if ps.is_complete],
            'nodes_to_delete': list(interior_nodes),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add to history
        self.operation_history.add_operation(operation)
        
        # Apply operation
        self._apply_single_operation(operation)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Deleted interior nodes: {len(interior_nodes)} nodes removed from {len(self.polygon_selectors)} polygon(s)")
        messagebox.showinfo("Success", f"Deleted {len(interior_nodes)} nodes")
    
    def highlight_interior(self):
        """Highlight nodes inside any polygon (union of all polygons)."""
        if not self._check_polygon_ready():
            return
        
        # Get nodes inside any polygon (union)
        interior_nodes = set()
        for polygon_selector in self.polygon_selectors:
            if polygon_selector.is_complete:
                nodes_in_polygon = polygon_selector.get_interior_nodes(self.current_positions)
                interior_nodes.update(nodes_in_polygon)
        
        if not interior_nodes:
            messagebox.showinfo("Info", "No nodes found inside polygons")
            return
        
        highlight_color = self.highlight_color_var.get()
        
        # Create operation
        import datetime
        operation = {
            'type': 'highlight',
            'polygons': [ps.vertices.copy() for ps in self.polygon_selectors if ps.is_complete],
            'nodes_to_highlight': list(interior_nodes),
            'highlight_color': highlight_color,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add to history
        self.operation_history.add_operation(operation)
        
        # Apply operation
        self._apply_single_operation(operation)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Highlighted interior nodes: {len(interior_nodes)} nodes set to {highlight_color} from {len(self.polygon_selectors)} polygon(s)")
        messagebox.showinfo("Success", f"Highlighted {len(interior_nodes)} nodes")
    
    def export_nodes(self):
        """Export list of nodes inside all polygons to file."""
        if not self._check_polygon_ready():
            return
        
        # Get nodes inside any polygon (union)
        interior_nodes = set()
        for polygon_selector in self.polygon_selectors:
            if polygon_selector.is_complete:
                nodes_in_polygon = polygon_selector.get_interior_nodes(self.current_positions)
                interior_nodes.update(nodes_in_polygon)
        
        if not interior_nodes:
            messagebox.showinfo("Info", "No nodes found inside polygons")
            return
        
        # Ask for file path
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.json'):
                import json
                data = {
                    'nodes': [list(node) for node in interior_nodes],
                    'positions': {str(node): list(self.current_positions[node]) 
                                 for node in interior_nodes},
                    'polygons': [ps.vertices for ps in self.polygon_selectors if ps.is_complete],
                    'num_polygons': len(self.polygon_selectors)
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                with open(filepath, 'w') as f:
                    f.write(f"Nodes inside {len(self.polygon_selectors)} polygon(s) ({len(interior_nodes)} total):\n\n")
                    for node in sorted(interior_nodes):
                        pos = self.current_positions[node]
                        f.write(f"{node}: ({pos[0]:.4f}, {pos[1]:.4f})\n")
            
            messagebox.showinfo("Success", f"Exported {len(interior_nodes)} nodes to {filepath}")
            print(f"Exported {len(interior_nodes)} nodes to {filepath}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def undo_operation(self):
        """Undo the last operation."""
        if not self.operation_history.can_undo():
            messagebox.showinfo("Info", "Nothing to undo")
            return
        
        # Undo in history
        undone_op = self.operation_history.undo()
        
        # Rebuild state from original + remaining history
        self._apply_operation_history()
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Undone operation: {undone_op['type']}")
        messagebox.showinfo("Undo", f"Undone {undone_op['type']} operation")
    
    def redo_operation(self):
        """Redo the previously undone operation."""
        if not self.operation_history.can_redo():
            messagebox.showinfo("Info", "Nothing to redo")
            return
        
        # Redo in history
        redone_op = self.operation_history.redo()
        
        # Apply the redone operation
        self._apply_single_operation(redone_op)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Redone operation: {redone_op['type']}")
        messagebox.showinfo("Redo", f"Redone {redone_op['type']} operation")
    
    def clear_polygon(self):
        """Clear all polygons from display."""
        for polygon_selector in self.polygon_selectors:
            polygon_selector.clear()
        
        self.polygon_selectors = []
        
        if self.active_polygon_selector:
            self.active_polygon_selector.clear()
            self.active_polygon_selector = None
        
        self.polygon_mode_active = False
        
        # Update status
        self.polygon_status_label.config(text="Status: All polygons cleared", foreground="gray")
        
        self.canvas.draw()
        print("All polygons cleared")
    
    def _check_polygon_ready(self) -> bool:
        """Check if at least one polygon is complete and ready for operations."""
        if self.current_graph is None:
            messagebox.showwarning("Warning", "Please generate a lattice first")
            return False
        
        if not self.polygon_selectors or not any(ps.is_complete for ps in self.polygon_selectors):
            messagebox.showwarning("Warning", "Please complete at least one polygon first")
            return False
        
        return True
    
    def _update_history_status(self):
        """Update history status label in toolbar."""
        current_ops = self.operation_history.get_current_state()
        total_ops = len(self.operation_history.operations)
        current_index = self.operation_history.current_index
        
        # Build status text
        status_parts = [f"Operations: {len(current_ops)}"]
        
        # Add undo/redo info
        can_undo = self.operation_history.can_undo()
        can_redo = self.operation_history.can_redo()
        
        if can_undo or can_redo:
            undo_redo = []
            if can_undo:
                undo_redo.append("Undo")
            if can_redo:
                redo_count = total_ops - current_index - 1
                undo_redo.append(f"Redo({redo_count})")
            status_parts.append(" | " + ", ".join(undo_redo))
        
        self.history_status_label.config(text="".join(status_parts))
        
        # Update debug info
        self._update_debug_info()
    
    def _update_debug_info(self):
        """Update debug information display."""
        if self.current_graph is None:
            self.debug_nodes_label.config(text="Nodes: -")
            self.debug_sublattice_label.config(text="N_A: -, N_B: -")
            self.debug_imbalance_label.config(text="Imbalance: -")
            return
        
        # Count nodes
        total_nodes = len(self.current_graph.nodes())
        
        # Count sublattices
        N_A = sum(1 for node in self.current_graph.nodes() if node[2] == 0)
        N_B = sum(1 for node in self.current_graph.nodes() if node[2] == 1)
        imbalance = abs(N_A - N_B)
        
        # Update labels
        self.debug_nodes_label.config(text=f"Nodes: {total_nodes}")
        self.debug_sublattice_label.config(text=f"N_A: {N_A}, N_B: {N_B}")
        self.debug_imbalance_label.config(text=f"Imbalance: {imbalance}")
    
    # ========== End Polygon Selection Methods ==========
    
    def _build_two_chains(self, chain1_type: str, chain2_type: str):
        """Build combination of any two chains."""
        x0_map = {
            'ns': self.x0_ns_var.get(),
            'swne': self.x0_swne_var.get(),
            'nwse': self.x0_nwse_var.get()
        }
        
        # Build first chain
        graph1, pos1, colors1 = LatticeFactory.create(
            chain1_type,
            self.config.lattice.num_units,
            self.config.lattice.bond_length,
            self.config.physics.bond_strength,
            x0=x0_map[chain1_type]
        )
        
        # Build second chain
        graph2, pos2, colors2 = LatticeFactory.create(
            chain2_type,
            self.config.lattice.num_units,
            self.config.lattice.bond_length,
            self.config.physics.bond_strength,
            x0=x0_map[chain2_type]
        )
        
        # Merge them
        merged_graph, merged_pos, merged_colors = LatticeModifier.merge_graphs(
            graph1, graph2,
            pos1, pos2,
            colors1, colors2
        )
        
        return merged_graph, merged_pos, merged_colors
    
    def _build_chain_with_chevron(self, chain_type: str):
        """Build a chain combined with its corresponding chevron orientation."""
        x0_map = {
            'ns': self.x0_ns_var.get(),
            'swne': self.x0_swne_var.get(),
            'nwse': self.x0_nwse_var.get()
        }
        
        # Build the chain
        chain_graph, chain_pos, chain_colors = LatticeFactory.create(
            chain_type,
            self.config.lattice.num_units,
            self.config.lattice.bond_length,
            self.config.physics.bond_strength,
            x0=x0_map[chain_type]
        )
        
        # Build chevron with matching orientation
        chevron_graph, chevron_pos, chevron_colors = LatticeFactory.create(
            'chevron',
            self.config.lattice.num_units,
            self.config.lattice.bond_length,
            self.config.physics.bond_strength,
            orientation=chain_type
        )
        
        # Merge them
        merged_graph, merged_pos, merged_colors = LatticeModifier.merge_graphs(
            chain_graph, chevron_graph,
            chain_pos, chevron_pos,
            chain_colors, chevron_colors
        )
        
        return merged_graph, merged_pos, merged_colors
    
    def _build_all_chains(self):
        """Build all three chains merged together."""
        # Build NS chain
        ns_graph, ns_pos, ns_colors = LatticeFactory.create(
            'ns',
            self.config.lattice.num_units,
            self.config.lattice.bond_length,
            self.config.physics.bond_strength,
            x0=self.x0_ns_var.get()
        )
        
        # Build SWNE chain
        swne_graph, swne_pos, swne_colors = LatticeFactory.create(
            'swne',
            self.config.lattice.num_units,
            self.config.lattice.bond_length,
            self.config.physics.bond_strength,
            x0=self.x0_swne_var.get()
        )
        
        # Build NWSE chain
        nwse_graph, nwse_pos, nwse_colors = LatticeFactory.create(
            'nwse',
            self.config.lattice.num_units,
            self.config.lattice.bond_length,
            self.config.physics.bond_strength,
            x0=self.x0_nwse_var.get()
        )
        
        # Merge NS and SWNE
        temp_graph, temp_pos, temp_colors = LatticeModifier.merge_graphs(
            ns_graph, swne_graph,
            ns_pos, swne_pos,
            ns_colors, swne_colors
        )
        
        # Merge with NWSE
        final_graph, final_pos, final_colors = LatticeModifier.merge_graphs(
            temp_graph, nwse_graph,
            temp_pos, nwse_pos,
            temp_colors, nwse_colors
        )
        
        return final_graph, final_pos, final_colors
    
    def draw_current_lattice(self):
        """Draw the current lattice."""
        if self.current_graph is None:
            return
        
        node_size = self.config.visualization.compute_node_size(self.config.lattice.num_units)
        
        visualizer = LatticeVisualizer(self.figure, self.ax)
        visualizer.draw_lattice(
            self.current_graph,
            self.current_positions,
            self.current_colors,
            node_size=node_size,
            edge_color=self.config.visualization.edge_color,
            title=f"{self.lattice_type_var.get().title()} Lattice"
        )
        
        self.canvas.draw()
        
        # Update debug info
        if hasattr(self, 'debug_nodes_label'):
            self._update_debug_info()
    
    def on_canvas_click(self, event):
        """Handle canvas click events."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        if self.vacancy_mode_var.get():
            self.remove_node_at(event.xdata, event.ydata)
        elif self.add_unit_mode_var.get():
            self.add_unit_at(event.xdata, event.ydata)
    
    def remove_node_at(self, x: float, y: float):
        """Remove node at clicked position and add to operation history."""
        removed = LatticeModifier.remove_node(
            self.current_graph,
            self.current_positions,
            self.current_colors,
            x, y
        )
        
        if removed:
            self.vacancies.append(removed)
            
            # Add to operation history
            import datetime
            operation = {
                'type': 'vacancy',
                'node': removed,
                'timestamp': datetime.datetime.now().isoformat()
            }
            self.operation_history.add_operation(operation)
            
            self.draw_current_lattice()
            print(f"Removed node: {removed}, Total vacancies: {len(self.vacancies)}")
    
    def add_unit_at(self, x: float, y: float):
        """Add unit cell at clicked position."""
        lattice_vectors = (self.config.lattice.bond_length * np.array([-np.sqrt(3)/2, 3/2]),
                          self.config.lattice.bond_length * np.array([np.sqrt(3)/2, 3/2]),
                          np.array([[0, 0], [0, -self.config.lattice.bond_length]]))
        
        clicked = LatticeModifier.add_unit_cell(
            self.current_graph,
            self.current_positions,
            self.current_colors,
            x, y,
            lattice_vectors,
            self.config.physics.bond_strength
        )
        
        if clicked:
            self.draw_current_lattice()
            print(f"Added unit cell at: {clicked}")
    
    def clear_vacancies(self):
        """Clear all vacancies and regenerate lattice."""
        self.vacancies = []
        self.generate_lattice()
    
    def open_zero_mode_window(self):
        """Open zero mode analysis window."""
        if self.current_graph is None:
            messagebox.showwarning("Warning", "Please generate a lattice first")
            return
        
        ZeroModeWindow(self.root, self)
    
    def analyze_zero_modes(self, plot_config: PlotConfig):
        """Perform zero mode analysis."""
        try:
            # Compute eigensystem
            H, eigenvals, eigenmodes = self.physics_engine.compute_eigensystem(self.current_graph)
            
            # Calculate sublattice imbalance for zero mode search
            # Zero modes = |N_A - N_B| where sub=0 is A, sub=1 is B
            N_A = sum(1 for node in self.current_graph.nodes() if node[2] == 0)
            N_B = sum(1 for node in self.current_graph.nodes() if node[2] == 1)
            sublattice_imbalance = abs(N_A - N_B)
            
            print(f"Sublattice count - A (red): {N_A}, B (black): {N_B}")
            print(f"Sublattice imbalance |N_A - N_B|: {sublattice_imbalance}")
            print(f"Total nodes in current graph: {len(self.current_graph.nodes())}")
            
            # Find zero modes based on sublattice imbalance
            zm_vecs = self.physics_engine.identify_zero_modes(
                eigenvals, eigenmodes, sublattice_imbalance
            )
            
            # Find zero modes based on sublattice imbalance
            zm_vecs = self.physics_engine.identify_zero_modes(
                eigenvals, eigenmodes, sublattice_imbalance
            )
            
            # Plot based on configuration
            if plot_config.plot_spectrum:
                fig = SpectrumVisualizer.plot_spectrum(eigenvals, "Energy Spectrum")
                fig.show()
            
            if not zm_vecs:
                messagebox.showinfo("Info", "No zero modes found")
                return
            
            # Plot zero modes
            if plot_config.plot_zm:
                self._plot_zero_modes(zm_vecs, plot_config)
            
            # Plot wavefunctions
            if plot_config.plot_wavefunction:
                self._plot_wavefunctions(zm_vecs, plot_config)
            
            # Plot interference
            if plot_config.plot_interference and 'interf' in zm_vecs:
                self._plot_interference(zm_vecs)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            print(f"Error: {e}")
    
    def _plot_zero_modes(self, zm_vecs: dict, plot_config: PlotConfig):
        """Plot zero mode probability distributions."""
        zm_indices = [k for k in zm_vecs.keys() if isinstance(k, int)]
        
        for idx in zm_indices:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            
            visualizer = LatticeVisualizer(fig, ax)
            visualizer.draw_zero_mode(
                self.current_graph,
                self.current_positions,
                zm_vecs[idx],
                title=f"Zero Mode {idx} of {len(zm_indices)}"
            )
            
            # Add line selection if requested
            if plot_config.select_lines:
                selector = InteractiveLineSelector(ax, plot_config.max_lines)
                lines = selector.select_lines()
                
                # Process projections
                if lines and plot_config.make_fit:
                    self._process_projections(zm_vecs[idx], lines, plot_config)
            
            fig.show()
    
    def _plot_wavefunctions(self, zm_vecs: dict, plot_config: PlotConfig):
        """Plot smooth wavefunctions."""
        wf_keys = [k for k in zm_vecs.keys() if isinstance(k, str) and 'ZM_wf' in k]
        
        for wf_key in wf_keys:
            wavefunction = zm_vecs[wf_key]
            positions = list(self.current_positions.values())
            
            fig, ax, im = WavefunctionVisualizer.plot_wavefunction(
                wavefunction,
                positions,
                plot_config.resolution_factor,
                plot_config.smoothing_factor,
                plot_config.sublattice_factor,
                title=f"Wavefunction {wf_key}"
            )
            
            # Add interactive colorbar control
            cbar = fig.axes[-1]  # Colorbar is last axes
            controller = ColorbarController(fig, im, cbar)
            
            fig.show()
    
    def _plot_interference(self, zm_vecs: dict):
        """Plot interference patterns."""
        for key in ['interf', 'interfM']:
            if key in zm_vecs:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                
                visualizer = LatticeVisualizer(fig, ax)
                visualizer.draw_zero_mode(
                    self.current_graph,
                    self.current_positions,
                    zm_vecs[key],
                    title=f"Zero Mode Interference ({key})"
                )
                
                fig.show()
    
    def _process_projections(self, zm_prob: np.ndarray, lines: list, plot_config: PlotConfig):
        """Process and plot projections along selected lines."""
        positions = list(self.current_positions.values())
        
        for line_start, line_end in lines:
            projections = self.physics_engine.project_eigenvector(
                zm_prob, line_start, line_end, positions
            )
            
            if projections:
                fig = ProjectionVisualizer.plot_projection(
                    projections,
                    "Zero Mode Projection",
                    plot_config.make_fit
                )
                fig.show()
    
    def save_config(self):
        """Save configuration to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.config.save(filepath)
                messagebox.showinfo("Success", "Configuration saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def load_config(self):
        """Load configuration from file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.config = ApplicationConfig.load(filepath)
                self._update_ui_from_config()
                messagebox.showinfo("Success", "Configuration loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {str(e)}")
    
    def _update_ui_from_config(self):
        """Update UI controls from config."""
        self.bond_length_var.set(self.config.lattice.bond_length)
        self.num_units_var.set(self.config.lattice.num_units)
        self.boundary_var.set(self.config.lattice.boundary)
        self.periodic_var.set(self.config.lattice.periodic_bc)
        self.zm_epsilon_var.set(f"{self.config.physics.zm_epsilon:.2e}")
    
    # ========== Session Save/Load Methods ==========
    
    def get_current_state(self) -> dict:
        """
        Collect all application state into dictionary.
        
        Returns:
            Complete state dictionary
        """
        state = {
            'lattice_type': self.lattice_type_var.get(),
            'lattice_config': StateSerializer.serialize_lattice_config(self.config.lattice)
        }
        
        # Add current graph if exists
        if self.current_graph is not None:
            state['current_graph'] = StateSerializer.serialize_graph(
                self.current_graph,
                self.current_positions,
                self.current_colors
            )
        
        # Add original graph if exists
        if self.original_graph is not None:
            state['original_graph'] = StateSerializer.serialize_graph(
                self.original_graph,
                self.original_positions,
                self.original_colors
            )
        
        # Add operation history
        state['operation_history'] = StateSerializer.serialize_operation_history(
            self.operation_history
        )
        
        # Add polygons
        state['polygons'] = StateSerializer.serialize_polygons(self.polygon_selectors)
        
        # Add chain positions
        state['chain_positions'] = {
            'x0_ns': self.x0_ns_var.get(),
            'x0_swne': self.x0_swne_var.get(),
            'x0_nwse': self.x0_nwse_var.get()
        }
        
        return state
    
    def restore_state(self, state: dict):
        """
        Restore application from state dictionary.
        
        Args:
            state: State dictionary to restore
        """
        try:
            # Restore lattice config
            if 'lattice_config' in state:
                config_data = state['lattice_config']
                self.config.lattice.bond_length = config_data.get('bond_length', 1.0)
                self.config.lattice.num_units = config_data.get('num_units', 7)
                self.config.lattice.boundary = config_data.get('boundary', 'bearded')
                self.config.lattice.periodic_bc = config_data.get('periodic_bc', False)
                self._update_ui_from_config()
            
            # Restore lattice type
            if 'lattice_type' in state:
                self.lattice_type_var.set(state['lattice_type'])
            
            # Restore chain positions
            if 'chain_positions' in state:
                self.x0_ns_var.set(state['chain_positions'].get('x0_ns', 0.0))
                self.x0_swne_var.set(state['chain_positions'].get('x0_swne', -3.5))
                self.x0_nwse_var.set(state['chain_positions'].get('x0_nwse', 3.5))
            
            # Restore current graph
            if 'current_graph' in state:
                self.current_graph, self.current_positions, self.current_colors = \
                    StateSerializer.deserialize_graph(state['current_graph'])
            
            # Restore original graph
            if 'original_graph' in state:
                self.original_graph, self.original_positions, self.original_colors = \
                    StateSerializer.deserialize_graph(state['original_graph'])
            
            # Restore operation history
            if 'operation_history' in state:
                history_data = state['operation_history']
                self.operation_history.operations = history_data['operations']
                self.operation_history.current_index = history_data['current_index']
            
            # Draw restored lattice
            if self.current_graph is not None:
                self.draw_current_lattice()
                self._update_history_status()
            
            print("State restored successfully")
            
        except Exception as e:
            print(f"Error restoring state: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_session(self, filepath: str = None):
        """
        Save complete session to file.
        
        Args:
            filepath: Optional filepath, otherwise asks user
        """
        if filepath is None:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".gsession",
                filetypes=[
                    ("Graphenery Session", "*.gsession"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                title="Save Session"
            )
        
        if not filepath:
            return
        
        try:
            state = self.get_current_state()
            StateSerializer.save_state(filepath, state, compress=False)
            messagebox.showinfo("Success", f"Session saved to:\n{filepath}")
            print(f"Session saved: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save session:\n{str(e)}")
            print(f"Save error: {e}")
            import traceback
            traceback.print_exc()
    
    def save_session_as(self):
        """Save session with file dialog."""
        self.save_session(filepath=None)
    
    def load_session(self, filepath: str = None):
        """
        Load session from file.
        
        Args:
            filepath: Optional filepath, otherwise asks user
        """
        if filepath is None:
            filepath = filedialog.askopenfilename(
                filetypes=[
                    ("Graphenery Session", "*.gsession"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                title="Load Session"
            )
        
        if not filepath:
            return
        
        try:
            state = StateSerializer.load_state(filepath)
            self.restore_state(state)
            messagebox.showinfo("Success", f"Session loaded from:\n{filepath}")
            print(f"Session loaded: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load session:\n{str(e)}")
            print(f"Load error: {e}")
            import traceback
            traceback.print_exc()
    
    def export_workflow(self):
        """Export operation workflow to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".gworkflow",
            filetypes=[
                ("Graphenery Workflow", "*.gworkflow"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ],
            title="Export Workflow"
        )
        
        if not filepath:
            return
        
        try:
            operations = self.operation_history.get_current_state()
            
            metadata = {
                'description': 'Graphenery operation workflow',
                'lattice_type': self.lattice_type_var.get(),
                'num_operations': len(operations)
            }
            
            WorkflowExporter.export_workflow(operations, filepath, metadata)
            messagebox.showinfo("Success", 
                              f"Exported {len(operations)} operations to:\n{filepath}")
            print(f"Workflow exported: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export workflow:\n{str(e)}")
            print(f"Export error: {e}")
    
    def import_workflow(self):
        """Import and apply workflow from file."""
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Graphenery Workflow", "*.gworkflow"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ],
            title="Import Workflow"
        )
        
        if not filepath:
            return
        
        try:
            operations, metadata = WorkflowExporter.import_workflow(filepath)
            
            if not operations:
                messagebox.showwarning("Warning", "No operations found in workflow file")
                return
            
            # Ask for confirmation
            num_ops = len(operations)
            response = messagebox.askyesno(
                "Import Workflow",
                f"This will apply {num_ops} operation(s) to the current lattice.\n\n"
                f"Continue?"
            )
            
            if not response:
                return
            
            # Apply operations
            for operation in operations:
                self.operation_history.add_operation(operation)
                self._apply_single_operation(operation)
            
            # Update display
            self.draw_current_lattice()
            self._update_history_status()
            
            messagebox.showinfo("Success", 
                              f"Applied {num_ops} operations from workflow")
            print(f"Workflow imported and applied: {num_ops} operations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import workflow:\n{str(e)}")
            print(f"Import error: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== End Session Save/Load Methods ==========
    
    def export_image(self):
        """Export current plot to image file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.figure.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", "Image exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def clear_cache(self):
        """Clear physics engine cache."""
        self.physics_engine.clear_cache()
        messagebox.showinfo("Success", "Cache cleared")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Graphenery
        
A graphene lattice visualization and analysis tool.

Features:
- Multiple lattice geometries
- Zero mode analysis
- Wavefunction visualization
- Interactive manipulation

Version 2.0 - Refactored Edition
"""
        messagebox.showinfo("About Graphenery", about_text)


class ZeroModeWindow:
    """Window for zero mode analysis options."""
    
    def __init__(self, parent, app: GrapheneryApp):
        """Initialize zero mode window."""
        self.app = app
        self.window = tk.Toplevel(parent)
        self.window.title("Zero Mode Analysis")
        self.window.geometry("400x600")
        
        self._create_controls()
    
    def _create_controls(self):
        """Create control widgets."""
        # Plot options
        options_frame = ttk.LabelFrame(self.window, text="Plot Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.plot_zm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Plot Zero Modes", 
                       variable=self.plot_zm_var).pack(anchor=tk.W, pady=2)
        
        self.plot_spectrum_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Plot Spectrum", 
                       variable=self.plot_spectrum_var).pack(anchor=tk.W, pady=2)
        
        self.plot_wf_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Plot Wavefunctions", 
                       variable=self.plot_wf_var).pack(anchor=tk.W, pady=2)
        
        self.plot_interf_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Plot Interference", 
                       variable=self.plot_interf_var).pack(anchor=tk.W, pady=2)
        
        # Line selection
        line_frame = ttk.LabelFrame(self.window, text="Line Selection", padding=10)
        line_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.select_lines_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(line_frame, text="Enable Line Selection", 
                       variable=self.select_lines_var).pack(anchor=tk.W, pady=2)
        
        ttk.Label(line_frame, text="Max Lines:").pack(anchor=tk.W)
        self.max_lines_var = tk.IntVar(value=1)
        ttk.Entry(line_frame, textvariable=self.max_lines_var, width=10).pack(anchor=tk.W, pady=2)
        
        self.make_fit_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(line_frame, text="Fit Exponential/Power Law", 
                       variable=self.make_fit_var).pack(anchor=tk.W, pady=2)
        
        # Wavefunction parameters
        wf_frame = ttk.LabelFrame(self.window, text="Wavefunction Options", padding=10)
        wf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(wf_frame, text="Resolution Factor:").pack(anchor=tk.W)
        self.res_factor_var = tk.IntVar(value=1)
        ttk.Entry(wf_frame, textvariable=self.res_factor_var, width=10).pack(anchor=tk.W, pady=2)
        
        ttk.Label(wf_frame, text="Smoothing Factor:").pack(anchor=tk.W)
        self.smooth_factor_var = tk.IntVar(value=1)
        ttk.Entry(wf_frame, textvariable=self.smooth_factor_var, width=10).pack(anchor=tk.W, pady=2)
        
        ttk.Label(wf_frame, text="Sublattice Factor:").pack(anchor=tk.W)
        self.sublat_factor_var = tk.DoubleVar(value=1.0)
        ttk.Entry(wf_frame, textvariable=self.sublat_factor_var, width=10).pack(anchor=tk.W, pady=2)
        
        # Run button
        ttk.Button(self.window, text="Run Analysis", 
                  command=self.run_analysis).pack(pady=20)
    
    def run_analysis(self):
        """Run zero mode analysis with current settings."""
        plot_config = PlotConfig(
            plot_zm=self.plot_zm_var.get(),
            plot_spectrum=self.plot_spectrum_var.get(),
            plot_wavefunction=self.plot_wf_var.get(),
            plot_interference=self.plot_interf_var.get(),
            select_lines=self.select_lines_var.get(),
            make_fit=self.make_fit_var.get(),
            max_lines=self.max_lines_var.get(),
            resolution_factor=self.res_factor_var.get(),
            smoothing_factor=self.smooth_factor_var.get(),
            sublattice_factor=self.sublat_factor_var.get()
        )
        
        self.app.analyze_zero_modes(plot_config)
        self.window.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = GrapheneryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()