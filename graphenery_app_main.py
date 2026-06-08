"""Main GUI application for Graphenery."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import networkx as nx

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
        
        # Auto-clear control (set when toolbar created)
        self.auto_clear_polygons_var = None
        
        # Graph editing system
        self.edit_mode_var = None  # Set when toolbar created
        self.edit_selected_nodes = []  # For bond add/remove (stores 2 nodes)
        self.edit_preview_line = None  # For bond preview
        
        # Subgraph management
        self.saved_subgraphs = []  # List of saved subgraphs
        # Each: {name, graph, positions, colors, original_graph}
        
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
        #self.root.geometry(f"{window_size}x{window_size}")
        # Let the OS pick a sensible initial size; user can resize freely
        self.root.geometry("")
        self.root.minsize(800, 600)
    
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
        
      #  # Left panel for controls with scrollbar
#        left_container = ttk.Frame(self.root, width=left_width)
#        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
#        left_container.pack_propagate(False)
#        
#        # Create canvas and scrollbar for scrollable left panel
#        canvas = tk.Canvas(left_container, width=canvas_width)

        left_container = ttk.Frame(self.root)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

         # Create canvas and scrollbar for scrollable left panel
        canvas = tk.Canvas(left_container, width=canvas_width*4)
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
        
        # Graph Editing Mode button  
        ttk.Button(frame, text="Graph Editing Mode", 
                  command=self.toggle_edit_toolbar,
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
        
        # Handle window close button - withdraw instead of destroy
        self.floating_toolbar.protocol("WM_DELETE_WINDOW", self.floating_toolbar.withdraw)
        
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
        
        # Separator
        ttk.Separator(ops_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Auto-clear checkbox
        self.auto_clear_polygons_var = tk.BooleanVar(value=True)
        auto_clear_check = ttk.Checkbutton(
            ops_frame,
            text="Auto-clear polygons after operation",
            variable=self.auto_clear_polygons_var
        )
        auto_clear_check.pack(fill=tk.X, pady=2)
        
        # Help text
        help_label = ttk.Label(
            ops_frame,
            text="(Enables sequential operations)",
            font=('Arial', 8),
            foreground='gray'
        )
        help_label.pack(pady=(0, 5))
        
        # Subgraph Management
        subgraph_frame = ttk.LabelFrame(main_frame, text="Subgraph Management", padding=10)
        subgraph_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(subgraph_frame, text="Save Current as Subgraph",
                  command=self.save_subgraph).pack(fill=tk.X, pady=2)
        ttk.Button(subgraph_frame, text="Merge Subgraphs",
                  command=self.open_merge_window).pack(fill=tk.X, pady=2)
        
        # Saved subgraphs counter
        self.saved_count_label = ttk.Label(subgraph_frame, 
                                          text="Saved: 0",
                                          font=('Arial', 8),
                                          foreground='gray')
        self.saved_count_label.pack(pady=2)
        
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
            # Convert from list to tuples (JSON stores as lists)
            nodes_to_keep = set(tuple(n) if isinstance(n, list) else n 
                              for n in operation['nodes_to_keep'])
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
            # Convert from list to tuples (JSON stores as lists)
            nodes_to_delete = [tuple(n) if isinstance(n, list) else n 
                             for n in operation['nodes_to_delete']]
            
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
            # Convert from list to tuples (JSON stores as lists)
            nodes_to_highlight = [tuple(n) if isinstance(n, list) else n 
                                for n in operation['nodes_to_highlight']]
            highlight_color = operation['highlight_color']
            
            for node in nodes_to_highlight:
                if node in self.current_colors:
                    self.current_colors[node] = highlight_color
            
            print(f"Highlight: Changed color of {len(nodes_to_highlight)} nodes to {highlight_color}")
        
        elif op_type == 'vacancy':
            # Single node removal (compatibility with existing vacancy system)
            node = operation['node']
            # Convert from list to tuple if needed
            if isinstance(node, list):
                node = tuple(node)
            if node in self.current_graph:
                self.current_graph.remove_node(node)
            if node in self.current_positions:
                del self.current_positions[node]
            if node in self.current_colors:
                del self.current_colors[node]
            
            print(f"Vacancy: Removed node {node}")
        
        elif op_type == 'add_bond':
            # Add bond between two nodes
            node1 = tuple(operation['node1'])
            node2 = tuple(operation['node2'])
            strength = operation['strength']
            
            if self.current_graph.has_node(node1) and self.current_graph.has_node(node2):
                self.current_graph.add_edge(node1, node2, weight=strength)
                print(f"Add bond: {node1} - {node2} (strength={strength})")
        
        elif op_type == 'remove_bond':
            # Remove bond between two nodes
            node1 = tuple(operation['node1'])
            node2 = tuple(operation['node2'])
            
            if self.current_graph.has_edge(node1, node2):
                self.current_graph.remove_edge(node1, node2)
                print(f"Remove bond: {node1} - {node2}")
        
        elif op_type == 'add_node':
            # Add a node
            node = tuple(operation['node'])
            position = np.array(operation['position'])
            color = operation['color']
            
            if node not in self.current_graph:
                self.current_graph.add_node(node)
                self.current_positions[node] = position
                self.current_colors[node] = color
                print(f"Add node: {node} at ({position[0]:.2f}, {position[1]:.2f})")
        
        elif op_type == 'remove_node':
            # Remove a node (graph editing version)
            node = tuple(operation['node'])
            
            if node in self.current_graph:
                self.current_graph.remove_node(node)
            if node in self.current_positions:
                del self.current_positions[node]
            if node in self.current_colors:
                del self.current_colors[node]
            
            print(f"Remove node: {node}")
    
    # ========== Polygon Selection Methods ==========
    
    def toggle_polygon_toolbar(self):
        """Toggle visibility of floating polygon toolbar."""
        try:
            if self.floating_toolbar.winfo_viewable():
                self.floating_toolbar.withdraw()
            else:
                self.floating_toolbar.deiconify()
                # Position near the plot
                plot_x = self.root.winfo_x() + self.root.winfo_width() - 300
                plot_y = self.root.winfo_y() + 100
                self.floating_toolbar.geometry(f"+{plot_x}+{plot_y}")
        except (tk.TclError, AttributeError):
            # Window was destroyed, recreate it
            self._create_floating_toolbar()
            self.floating_toolbar.deiconify()
    
    def toggle_edit_toolbar(self):
        """Toggle visibility of graph editing toolbar."""
        # Check if toolbar exists and is not destroyed
        if not hasattr(self, 'edit_toolbar') or not self.edit_toolbar.winfo_exists():
            self._create_edit_toolbar()
            return
        
        # Toggle visibility
        try:
            if self.edit_toolbar.winfo_viewable():
                self.edit_toolbar.withdraw()
            else:
                self.edit_toolbar.deiconify()
                # Position next to polygon toolbar
                plot_x = self.root.winfo_x() + self.root.winfo_width() - 600
                plot_y = self.root.winfo_y() + 100
                self.edit_toolbar.geometry(f"+{plot_x}+{plot_y}")
        except tk.TclError:
            # Window was destroyed, recreate it
            self._create_edit_toolbar()
    
    def _create_edit_toolbar(self):
        """Create floating toolbar for graph editing."""
        self.edit_toolbar = tk.Toplevel(self.root)
        self.edit_toolbar.title("Graph Editing")
        self.edit_toolbar.geometry("280x300")
        self.edit_toolbar.resizable(True, True)
        
        # Make it stay on top but allow dragging
        self.edit_toolbar.attributes('-topmost', True)
        
        # Handle window close button - withdraw instead of destroy
        self.edit_toolbar.protocol("WM_DELETE_WINDOW", self.edit_toolbar.withdraw)
        
        # Main frame
        main_frame = ttk.Frame(self.edit_toolbar, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Graph Editing", 
                         font=('Arial', 12, 'bold'))
        title.pack(pady=5)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Edit Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.edit_mode_var = tk.StringVar(value="none")
        ttk.Radiobutton(mode_frame, text="Add Bond", 
                       variable=self.edit_mode_var, 
                       value="add_bond").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Remove Bond", 
                       variable=self.edit_mode_var, 
                       value="remove_bond").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Add Node", 
                       variable=self.edit_mode_var, 
                       value="add_node").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Remove Node", 
                       variable=self.edit_mode_var, 
                       value="remove_node").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="None (Off)", 
                       variable=self.edit_mode_var, 
                       value="none").pack(anchor=tk.W)
        
        # Status label
        self.edit_status_label = ttk.Label(main_frame, 
                                          text="Edit mode: Off",
                                          foreground="gray",
                                          wraplength=250)
        self.edit_status_label.pack(pady=10)
        
        # Trace mode changes
        self.edit_mode_var.trace('w', self._on_edit_mode_change)
        
        # Instructions
        instructions = ttk.Label(main_frame,
                               text="Bond: Click 2 nodes\nNode: Opens dialog",
                               font=('Arial', 8),
                               foreground='gray')
        instructions.pack(pady=5)
    
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
        
        # Auto-clear polygons if enabled
        num_polygons = len([ps for ps in self.polygon_selectors if ps.is_complete])
        if hasattr(self, 'auto_clear_polygons_var') and self.auto_clear_polygons_var and self.auto_clear_polygons_var.get():
            self.clear_polygon()
            clear_msg = " (polygons cleared)"
        else:
            clear_msg = " (polygons kept)"
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Extracted subgraph: {len(interior_nodes)} nodes from {num_polygons} polygon(s){clear_msg}")
        messagebox.showinfo("Success", f"Extracted {len(interior_nodes)} nodes")
    
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
        
        # Auto-clear polygons if enabled
        num_polygons = len([ps for ps in self.polygon_selectors if ps.is_complete])
        if hasattr(self, 'auto_clear_polygons_var') and self.auto_clear_polygons_var and self.auto_clear_polygons_var.get():
            self.clear_polygon()
            clear_msg = " (polygons cleared)"
        else:
            clear_msg = " (polygons kept)"
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Deleted interior: {len(interior_nodes)} nodes from {num_polygons} polygon(s){clear_msg}")
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
        
        # Auto-clear polygons if enabled
        num_polygons = len([ps for ps in self.polygon_selectors if ps.is_complete])
        if hasattr(self, 'auto_clear_polygons_var') and self.auto_clear_polygons_var and self.auto_clear_polygons_var.get():
            self.clear_polygon()
            clear_msg = " (polygons cleared)"
        else:
            clear_msg = " (polygons kept)"
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Highlighted interior: {len(interior_nodes)} nodes to {highlight_color} from {num_polygons} polygon(s){clear_msg}")
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
    
    def _restore_polygons_from_operation(self, operation):
        """
        Restore polygon display from operation data (for undo).
        
        Args:
            operation: Operation dict containing 'polygons' key
        """
        if 'polygons' not in operation or not operation['polygons']:
            return
        
        # Clear current polygons first
        for ps in self.polygon_selectors:
            if hasattr(ps, 'clear'):
                ps.clear()
        self.polygon_selectors = []
        
        # Recreate polygons from operation data
        from polygon_selector import PolygonSelector
        
        for poly_vertices in operation['polygons']:
            # Create new polygon selector
            ps = PolygonSelector(
                self.ax,
                color='blue',  # Restored polygons in blue (visual feedback)
                linewidth=2.0
            )
            
            # Set vertices and mark as complete
            ps.vertices = poly_vertices
            ps.is_complete = True
            
            # Draw the polygon
            if hasattr(ps, '_draw_polygon'):
                ps._draw_polygon()
            
            # Add to list
            self.polygon_selectors.append(ps)
        
        # Update status
        if hasattr(self, 'polygon_status_label'):
            num_restored = len(operation['polygons'])
            self.polygon_status_label.config(
                text=f"Status: {num_restored} polygon(s) restored",
                foreground="blue"
            )
        
        # Redraw canvas
        self.canvas.draw()
        
        print(f"Restored {len(operation['polygons'])} polygon(s) from undo")
    
    def undo_operation(self):
        """Undo the last operation."""
        if not self.operation_history.can_undo():
            messagebox.showinfo("Info", "Nothing to undo")
            return
        
        # Undo in history
        undone_op = self.operation_history.undo()
        
        # Rebuild state from original + remaining history
        self._apply_operation_history()
        
        # Restore polygons from undone operation
        self._restore_polygons_from_operation(undone_op)
        
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
        
        # Clear polygons if auto-clear is enabled
        if hasattr(self, 'auto_clear_polygons_var') and self.auto_clear_polygons_var and self.auto_clear_polygons_var.get():
            self.clear_polygon()
        
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
    
    # ========== Graph Editing Methods ==========
    
    def _on_edit_mode_change(self, *args):
        """Handle edit mode changes."""
        mode = self.edit_mode_var.get()
        
        # Clear any previous selection
        self._clear_edit_selection()
        
        # Update status
        mode_labels = {
            'add_bond': 'Add Bond: Click 2 nodes',
            'remove_bond': 'Remove Bond: Click 2 nodes',
            'add_node': 'Add Node: Choose method',
            'remove_node': 'Remove Node: Click node',
            'none': 'Edit mode: Off'
        }
        
        label_text = mode_labels.get(mode, 'Edit mode: Off')
        color = 'green' if mode != 'none' else 'gray'
        self.edit_status_label.config(text=label_text, foreground=color)
        
        # If add_node mode, open dialog immediately
        # Don't reset mode - let user add multiple nodes
        if mode == 'add_node':
            self._add_node_dialog()
    
    def _clear_edit_selection(self):
        """Clear edit mode selection and preview."""
        self.edit_selected_nodes = []
        
        # Remove preview line if exists
        if self.edit_preview_line:
            self.edit_preview_line.remove()
            self.edit_preview_line = None
            self.canvas.draw()
    
    def _on_node_click_edit(self, node):
        """Handle node click in edit mode."""
        mode = self.edit_mode_var.get()
        
        if mode not in ['add_bond', 'remove_bond']:
            return
        
        # Add to selection
        self.edit_selected_nodes.append(node)
        
        if len(self.edit_selected_nodes) == 1:
            # First node - highlight green
            self._highlight_edit_node(node, 'green')
            self.edit_status_label.config(text=f"Selected node 1: {node}")
            
        elif len(self.edit_selected_nodes) == 2:
            # Second node - highlight blue and process
            self._highlight_edit_node(node, 'blue')
            self.edit_status_label.config(text=f"Selected node 2: {node}")
            
            # Show preview
            self._show_bond_preview()
            
            # Process based on mode
            if mode == 'add_bond':
                self._add_bond_dialog()
            elif mode == 'remove_bond':
                self._remove_bond()
            
            # Reset for next operation
            self.root.after(1000, self._clear_edit_selection)
    
    def _highlight_edit_node(self, node, color):
        """Highlight a selected node."""
        if node in self.current_positions:
            pos = self.current_positions[node]
            self.ax.plot(pos[0], pos[1], 'o', color=color, 
                        markersize=15, alpha=0.5, zorder=10)
            self.canvas.draw()
    
    def _show_bond_preview(self):
        """Show preview line between two selected nodes."""
        if len(self.edit_selected_nodes) != 2:
            return
        
        node1, node2 = self.edit_selected_nodes
        if node1 in self.current_positions and node2 in self.current_positions:
            pos1 = self.current_positions[node1]
            pos2 = self.current_positions[node2]
            
            self.edit_preview_line, = self.ax.plot(
                [pos1[0], pos2[0]], 
                [pos1[1], pos2[1]], 
                'g--', linewidth=2, alpha=0.7, zorder=5
            )
            self.canvas.draw()
    
    def _add_bond_dialog(self):
        """Show dialog to add bond with strength input."""
        if len(self.edit_selected_nodes) != 2:
            return
        
        node1, node2 = self.edit_selected_nodes
        
        # Check if bond already exists
        if self.current_graph.has_edge(node1, node2):
            messagebox.showwarning("Bond Exists", 
                                  f"Bond already exists between {node1} and {node2}")
            return
        
        # Dialog for bond strength
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Bond")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text=f"Add bond between:").pack(pady=5)
        ttk.Label(dialog, text=f"{node1} and {node2}").pack()
        
        ttk.Label(dialog, text="Bond strength:").pack(pady=5)
        strength_var = tk.DoubleVar(value=1.0)
        ttk.Entry(dialog, textvariable=strength_var).pack()
        
        def confirm():
            strength = strength_var.get()
            self._add_bond(node1, node2, strength)
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Add", command=confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
    
    def _add_bond(self, node1, node2, strength=1.0):
        """Add bond between two nodes."""
        if not self.current_graph.has_node(node1) or not self.current_graph.has_node(node2):
            messagebox.showerror("Error", "One or both nodes don't exist")
            return
        
        # Add edge
        self.current_graph.add_edge(node1, node2, weight=strength)
        
        # Add to operation history
        import datetime
        operation = {
            'type': 'add_bond',
            'node1': list(node1),  # Convert tuple to list for JSON
            'node2': list(node2),
            'strength': strength,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.operation_history.add_operation(operation)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Added bond: {node1} - {node2} (strength={strength})")
        messagebox.showinfo("Success", f"Bond added with strength {strength}")
    
    def _remove_bond(self):
        """Remove bond between two selected nodes."""
        if len(self.edit_selected_nodes) != 2:
            return
        
        node1, node2 = self.edit_selected_nodes
        
        if not self.current_graph.has_edge(node1, node2):
            messagebox.showwarning("No Bond", 
                                  f"No bond exists between {node1} and {node2}")
            return
        
        # Save bond strength before removing
        strength = self.current_graph[node1][node2].get('weight', 1.0)
        
        # Remove edge
        self.current_graph.remove_edge(node1, node2)
        
        # Add to operation history
        import datetime
        operation = {
            'type': 'remove_bond',
            'node1': list(node1),
            'node2': list(node2),
            'strength': strength,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.operation_history.add_operation(operation)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Removed bond: {node1} - {node2}")
        messagebox.showinfo("Success", "Bond removed")
    
    def _add_node_dialog(self):
        """Show dialog to add a new node."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Node")
        dialog.geometry("400x500")
        
        # Coordinates frame
        coord_frame = ttk.LabelFrame(dialog, text="Node Position", padding=10)
        coord_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(coord_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        x_var = tk.DoubleVar(value=0.0)
        x_entry = ttk.Entry(coord_frame, textvariable=x_var, width=12)
        x_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(coord_frame, text="Y:").grid(row=0, column=2, padx=(10, 0), sticky=tk.W)
        y_var = tk.DoubleVar(value=0.0)
        y_entry = ttk.Entry(coord_frame, textvariable=y_var, width=12)
        y_entry.grid(row=0, column=3, padx=5)
        
        # Click to set coordinates button
        def set_by_click():
            dialog.withdraw()  # Hide dialog
            messagebox.showinfo("Click Mode", 
                              "Click on the canvas to set coordinates\n"
                              "Dialog will reappear with coordinates filled")
            
            # Set up one-time click handler
            def on_click_for_coords(event):
                if event.inaxes == self.ax and event.xdata is not None:
                    x_var.set(round(event.xdata, 3))
                    y_var.set(round(event.ydata, 3))
                    update_calculated_index()  # Update index display
                    dialog.deiconify()  # Show dialog again
                    self.canvas.mpl_disconnect(cid)
            
            cid = self.canvas.mpl_connect('button_press_event', on_click_for_coords)
        
        ttk.Button(coord_frame, text="📍 Click to Set", 
                  command=set_by_click).grid(row=1, column=0, columnspan=4, pady=5)
        
        # Snap to grid option
        snap_var = tk.BooleanVar(value=False)
        snap_check = ttk.Checkbutton(coord_frame, text="☑ Snap to nearest lattice index", 
                                     variable=snap_var,
                                     command=lambda: toggle_index_frame())
        snap_check.grid(row=2, column=0, columnspan=4, sticky=tk.W)
        
        # Lattice index frame (shown when snap is checked)
        index_frame = ttk.LabelFrame(dialog, text="Calculated Lattice Index", padding=10)
        
        ttk.Label(index_frame, text="i:").grid(row=0, column=0, sticky=tk.W)
        i_var = tk.IntVar(value=0)
        i_entry = ttk.Entry(index_frame, textvariable=i_var, width=8)
        i_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(index_frame, text="j:").grid(row=0, column=2, padx=(10, 0), sticky=tk.W)
        j_var = tk.IntVar(value=0)
        j_entry = ttk.Entry(index_frame, textvariable=j_var, width=8)
        j_entry.grid(row=0, column=3, padx=5)
        
        ttk.Label(index_frame, text="(Editable before confirming)", 
                 font=('Arial', 8), foreground='gray').grid(row=1, column=0, columnspan=4, pady=5)
        
        def update_calculated_index(*args):
            """Calculate and display lattice indices from coordinates."""
            if snap_var.get():
                try:
                    x, y = x_var.get(), y_var.get()
                    bond_length = self.config.lattice.bond_length
                    
                    # Lattice vectors (forward transform):
                    # pos = i * a1 + j * a2
                    #a1 = [-√3/2 * b, 3/2 * b]   # Points up-left (120°)
                    #a2 = [√3/2 * b, 3/2 * b]    # Points up-right (60°)
                    # Inverse transform to find (i, j) from (x, y):
                    #x = (j - i) * √3/2 * b
                    #y = (i + j) * 3/2 * b
                    #Solving:
                    #i = y/(3b) - x/(√3*b)
                    #j = y/(3b) + x/(√3*b)
                    i = y / (3 * bond_length) - x / (np.sqrt(3) * bond_length)
                    j = y / (3 * bond_length) + x / (np.sqrt(3) * bond_length)
                    
                    i_var.set(int(round(i)))
                    j_var.set(int(round(j)))
                except:
                    pass
        
        def toggle_index_frame():
            """Show/hide index frame based on snap checkbox."""
            if snap_var.get():
                index_frame.pack(fill=tk.X, padx=10, pady=5)
                update_calculated_index()
            else:
                index_frame.pack_forget()
        
        # Trace coordinate changes to update index
        x_var.trace('w', update_calculated_index)
        y_var.trace('w', update_calculated_index)
        
        # Sublattice selection
        sub_frame = ttk.LabelFrame(dialog, text="Sublattice", padding=10)
        sub_frame.pack(fill=tk.X, padx=10, pady=5)
        
        sub_var = tk.IntVar(value=0)
        ttk.Radiobutton(sub_frame, text="A (0, red)", 
                       variable=sub_var, value=0).pack(anchor=tk.W)
        ttk.Radiobutton(sub_frame, text="B (1, black)", 
                       variable=sub_var, value=1).pack(anchor=tk.W)
        
        # Color selection
        color_frame = ttk.LabelFrame(dialog, text="Node Color", padding=10)
        color_frame.pack(fill=tk.X, padx=10, pady=5)
        
        use_default_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(color_frame, text="Use default sublattice color", 
                       variable=use_default_var).pack(anchor=tk.W)
        
        custom_frame = ttk.Frame(color_frame)
        custom_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(custom_frame, text="Custom RGB:").grid(row=0, column=0)
        r_var = tk.IntVar(value=255)
        g_var = tk.IntVar(value=0)
        b_var = tk.IntVar(value=0)
        
        ttk.Label(custom_frame, text="R:").grid(row=0, column=1)
        ttk.Spinbox(custom_frame, from_=0, to=255, textvariable=r_var, width=5).grid(row=0, column=2)
        ttk.Label(custom_frame, text="G:").grid(row=0, column=3)
        ttk.Spinbox(custom_frame, from_=0, to=255, textvariable=g_var, width=5).grid(row=0, column=4)
        ttk.Label(custom_frame, text="B:").grid(row=0, column=5)
        ttk.Spinbox(custom_frame, from_=0, to=255, textvariable=b_var, width=5).grid(row=0, column=6)
        
        def confirm():
            sublattice = sub_var.get()
            snap = snap_var.get()
            
            # Determine color
            if use_default_var.get():
                color = 'red' if sublattice == 0 else 'black'
            else:
                r, g, b = r_var.get(), g_var.get(), b_var.get()
                color = f'#{r:02x}{g:02x}{b:02x}'
            
            dialog.destroy()
            
            if snap:
                # Use edited lattice indices
                i, j = i_var.get(), j_var.get()
                self._add_node_by_index(i, j, sublattice, color)
            else:
                # Use exact coordinates
                x, y = x_var.get(), y_var.get()
                self._add_node_at_position(x, y, sublattice, color)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Add Node", command=confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _add_node_snapped(self, x, y, sublattice, color):
        """Add node snapped to nearest lattice index."""
        bond_length = self.config.lattice.bond_length
        
        # Calculate nearest lattice indices
        i = int(round(x / (1.5 * bond_length)))
        j = int(round(y / (np.sqrt(3) * bond_length)))
        
        # Use the index-based method which calculates exact position
        self._add_node_by_index(i, j, sublattice, color)
    
    def _add_node_at_position(self, x, y, sublattice, color):
        """Add node at specific 2D coordinates (not snapped to lattice)."""
        # For coordinate-based nodes (not snapped to grid), use custom node format
        # to avoid collisions with lattice nodes
        # Format: ('custom', unique_id, sublattice)
        
        # Find next available custom ID
        existing_custom_ids = [node[1] for node in self.current_graph.nodes() 
                              if isinstance(node, tuple) and len(node) == 3 
                              and node[0] == 'custom']
        if existing_custom_ids:
            next_id = max(existing_custom_ids) + 1
        else:
            next_id = 0
        
        node = ('custom', next_id, sublattice)
        
        # Add node
        self.current_graph.add_node(node)
        self.current_positions[node] = np.array([x, y])
        self.current_colors[node] = color
        
        # Add to operation history
        import datetime
        operation = {
            'type': 'add_node',
            'node': list(node),
            'position': [x, y],
            'color': color,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.operation_history.add_operation(operation)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Added node: {node} at ({x:.2f}, {y:.2f})")
        messagebox.showinfo("Success", f"Added custom node at ({x:.2f}, {y:.2f})")
    
    def _add_node_by_index(self, i, j, sublattice, color):
        """Add node by lattice index."""
        node = (i, j, sublattice)
        
        # Check if node already exists
        if node in self.current_graph.nodes():
            messagebox.showwarning("Node Exists", f"Node {node} already exists")
            return
        
        # Calculate position from lattice indices
        bond_length = self.config.lattice.bond_length
        #a1 = np.array([1.5 * bond_length, np.sqrt(3)/2 * bond_length])
        #a2 = np.array([0, np.sqrt(3) * bond_length]
        a1 = np.array([-np.sqrt(3)/2 * bond_length, 3/2 * bond_length])
        a2 = np.array([np.sqrt(3)/2 * bond_length, 3/2 * bond_length])
        
        # Base position
        pos = i * a1 + j * a2
        
        # Offset for sublattice
        if sublattice == 1:  # B sublattice
            #pos += np.array([bond_length, 0])
            pos += np.array([0,-bond_length])
        
        # Add node
        self.current_graph.add_node(node)
        self.current_positions[node] = pos
        self.current_colors[node] = color
        
        # Add to operation history
        import datetime
        operation = {
            'type': 'add_node',
            'node': list(node),
            'position': [float(pos[0]), float(pos[1])],
            'color': color,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.operation_history.add_operation(operation)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Added node: {node} at position ({pos[0]:.2f}, {pos[1]:.2f})")
        messagebox.showinfo("Success", f"Added node {node}")
    
    def _remove_node_edit(self, node):
        """Remove a node (graph editing version with undo/redo support)."""
        if not self.current_graph.has_node(node):
            messagebox.showwarning("No Node", f"Node {node} doesn't exist")
            return
        
        # Save node data for undo
        position = self.current_positions[node].copy()
        color = self.current_colors[node]
        
        # Save connected edges and their weights for undo
        edges = []
        for neighbor in self.current_graph.neighbors(node):
            weight = self.current_graph[node][neighbor].get('weight', 1.0)
            edges.append((list(node), list(neighbor), weight))
        
        # Remove node
        self.current_graph.remove_node(node)
        if node in self.current_positions:
            del self.current_positions[node]
        if node in self.current_colors:
            del self.current_colors[node]
        
        # Add to operation history
        import datetime
        operation = {
            'type': 'remove_node',
            'node': list(node),
            'position': [float(position[0]), float(position[1])],
            'color': color,
            'edges': edges,  # For potential undo support
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.operation_history.add_operation(operation)
        
        # Update display
        self.draw_current_lattice()
        self._update_history_status()
        
        print(f"Removed node: {node}")
        messagebox.showinfo("Success", f"Removed node {node}")
    
    # ========== End Graph Editing Methods ==========
    
    # ========== Subgraph Management Methods ==========
    
    def save_subgraph(self):
        """Save current graph as a subgraph to memory."""
        if self.current_graph is None or len(self.current_graph.nodes()) == 0:
            messagebox.showwarning("Warning", "No graph to save")
            return
        
        # Dialog for naming
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Subgraph")
        dialog.geometry("350x150")
        
        ttk.Label(dialog, text="Subgraph Name:").pack(pady=10)
        
        # Default name with running index
        default_name = f"subgraph_{len(self.saved_subgraphs)}"
        name_var = tk.StringVar(value=default_name)
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.pack(pady=5)
        name_entry.select_range(0, tk.END)
        name_entry.focus()
        
        info_label = ttk.Label(dialog, 
                              text=f"Nodes: {len(self.current_graph.nodes())}, "
                                   f"Edges: {len(self.current_graph.edges())}",
                              font=('Arial', 9),
                              foreground='gray')
        info_label.pack(pady=5)
        
        def confirm():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Name cannot be empty")
                return
            
            # Check for duplicate names
            if any(sg['name'] == name for sg in self.saved_subgraphs):
                if not messagebox.askyesno("Duplicate Name", 
                                          f"Subgraph '{name}' already exists. Overwrite?"):
                    return
                # Remove old one
                self.saved_subgraphs = [sg for sg in self.saved_subgraphs if sg['name'] != name]
            
            # Save subgraph
            subgraph_data = {
                'name': name,
                'graph': self.current_graph.copy(),
                'positions': self.current_positions.copy(),
                'colors': self.current_colors.copy(),
                'original_graph': self.original_graph.copy() if hasattr(self, 'original_graph') else None,
                'node_count': len(self.current_graph.nodes()),
                'edge_count': len(self.current_graph.edges())
            }
            
            self.saved_subgraphs.append(subgraph_data)
            
            # Update counter
            if hasattr(self, 'saved_count_label'):
                self.saved_count_label.config(text=f"Saved: {len(self.saved_subgraphs)}")
            
            dialog.destroy()
            print(f"Saved subgraph '{name}' with {subgraph_data['node_count']} nodes")
            messagebox.showinfo("Success", f"Saved subgraph '{name}'")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save", command=confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Enter key to confirm
        dialog.bind('<Return>', lambda e: confirm())
    
    def open_merge_window(self):
        """Open window for selecting and merging subgraphs."""
        if len(self.saved_subgraphs) == 0:
            messagebox.showinfo("Info", "No saved subgraphs to merge")
            return
        
        # Create merge window
        merge_window = tk.Toplevel(self.root)
        merge_window.title("Merge Subgraphs")
        
        # Use simpler interface (list-only) for better mobile performance
        USE_THUMBNAILS = False  # Set to True for desktop, False for mobile
        
        if USE_THUMBNAILS:
            merge_window.geometry("900x600")
        else:
            merge_window.geometry("450x600")
        
        # Title
        title = ttk.Label(merge_window, text="Select Subgraphs to Merge", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Main container with two panels
        main_container = ttk.Frame(merge_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - List
        left_panel = ttk.LabelFrame(main_container, text="Subgraph List", padding=10)
        if USE_THUMBNAILS:
            left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
            left_panel.config(width=250)
        else:
            left_panel.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Thumbnails (only if enabled)
        if USE_THUMBNAILS:
            right_panel = ttk.LabelFrame(main_container, text="Visual Selection", padding=10)
            right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Selection tracking
        selected_indices = set()
        list_checkboxes = []
        thumbnail_frames = []
        
        # Populate list panel
        list_canvas = tk.Canvas(left_panel, width=220)
        list_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=list_canvas.yview)
        list_frame = ttk.Frame(list_canvas)
        
        list_canvas.create_window((0, 0), window=list_frame, anchor="nw")
        list_canvas.configure(yscrollcommand=list_scrollbar.set)
        
        list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def toggle_selection(idx):
            """Toggle selection of a subgraph."""
            if idx in selected_indices:
                selected_indices.remove(idx)
            else:
                selected_indices.add(idx)
            update_visuals()
        
        def update_visuals():
            """Update visual indication of selection."""
            # Update checkboxes
            for i, var in enumerate(list_checkboxes):
                var.set(i in selected_indices)
            
            # Update thumbnail highlights with background color
            for i, frame in enumerate(thumbnail_frames):
                if i in selected_indices:
                    frame.config(relief=tk.SOLID, borderwidth=3)
                else:
                    frame.config(relief=tk.RAISED, borderwidth=1)
        
        # Create list items
        for i, sg in enumerate(self.saved_subgraphs):
            item_frame = ttk.Frame(list_frame)
            item_frame.pack(fill=tk.X, pady=2)
            
            var = tk.BooleanVar(value=False)
            list_checkboxes.append(var)
            
            cb = ttk.Checkbutton(item_frame, text=sg['name'], variable=var,
                                command=lambda idx=i: toggle_selection(idx))
            cb.pack(side=tk.LEFT)
            
            info = ttk.Label(item_frame, 
                           text=f"({sg['node_count']}n, {sg['edge_count']}e)",
                           font=('Arial', 8),
                           foreground='gray')
            info.pack(side=tk.RIGHT)
        
        list_frame.update_idletasks()
        list_canvas.config(scrollregion=list_canvas.bbox("all"))
        
        # Populate thumbnail panel (only if enabled)
        if USE_THUMBNAILS:
            thumb_canvas = tk.Canvas(right_panel)
            thumb_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=thumb_canvas.yview)
            thumb_frame = ttk.Frame(thumb_canvas)
            
            thumb_canvas.create_window((0, 0), window=thumb_frame, anchor="nw")
            thumb_canvas.configure(yscrollcommand=thumb_scrollbar.set)
            
            thumb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            thumb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create thumbnails (3 per row) - Lightweight text-based version
            for i, sg in enumerate(self.saved_subgraphs):
                # Container for each thumbnail
                row = i // 3
                col = i % 3
                
                container = ttk.Frame(thumb_frame, relief=tk.RAISED, borderwidth=1, padding=10)
                container.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                thumbnail_frames.append(container)
                
                # Make clickable
                def make_click_handler(idx):
                    return lambda e: toggle_selection(idx)
                
                container.bind('<Button-1>', make_click_handler(i))
                
                # Name label (large)
                name_label = ttk.Label(container, text=sg['name'], 
                                      font=('Arial', 12, 'bold'),
                                      foreground='blue')
                name_label.pack(pady=5)
                name_label.bind('<Button-1>', make_click_handler(i))
                
                # Info display
                info_text = (f"Nodes: {sg['node_count']}\n"
                            f"Edges: {sg['edge_count']}")
                info_label = ttk.Label(container, text=info_text,
                                      font=('Arial', 10),
                                      justify=tk.CENTER)
                info_label.pack(pady=5)
                info_label.bind('<Button-1>', make_click_handler(i))
                
                # Visual indicator box (colored frame)
                color_frame = tk.Frame(container, width=80, height=80, 
                                      bg='lightblue', relief=tk.SUNKEN, borderwidth=2)
                color_frame.pack(pady=5)
                color_frame.pack_propagate(False)
                color_frame.bind('<Button-1>', make_click_handler(i))
                
                # Node count in box
                count_label = tk.Label(color_frame, 
                                      text=f"{sg['node_count']}\nnodes",
                                      font=('Arial', 14, 'bold'),
                                      bg='lightblue',
                                      fg='darkblue')
                count_label.pack(expand=True)
                count_label.bind('<Button-1>', make_click_handler(i))
            
            # Configure grid weights
            for j in range(3):
                thumb_frame.columnconfigure(j, weight=1)
            
            thumb_frame.update_idletasks()
            thumb_canvas.config(scrollregion=thumb_canvas.bbox("all"))
        
        # Bottom panel - Merge options and confirm
        bottom_panel = ttk.Frame(merge_window, padding=10)
        bottom_panel.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Merge options
        options_frame = ttk.LabelFrame(bottom_panel, text="Merge Options", padding=10)
        options_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Duplicate detection method
        dup_method_var = tk.StringVar(value="index")
        ttk.Radiobutton(options_frame, text="Detect duplicates by lattice index (i,j,sub)", 
                       variable=dup_method_var, value="index").pack(anchor=tk.W)
        
        ttk.Radiobutton(options_frame, text="Detect duplicates by position tolerance:", 
                       variable=dup_method_var, value="tolerance").pack(anchor=tk.W)
        
        tol_frame = ttk.Frame(options_frame)
        tol_frame.pack(anchor=tk.W, padx=20)
        ttk.Label(tol_frame, text="Tolerance:").pack(side=tk.LEFT)
        tolerance_var = tk.DoubleVar(value=0.01)
        ttk.Entry(tol_frame, textvariable=tolerance_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(tol_frame, text="Å").pack(side=tk.LEFT)
        
        # Button panel
        button_panel = ttk.Frame(bottom_panel)
        button_panel.pack(side=tk.RIGHT)
        
        def confirm_merge():
            """Perform the merge."""
            if len(selected_indices) == 0:
                messagebox.showwarning("Warning", "No subgraphs selected")
                return
            
            selected = [self.saved_subgraphs[i] for i in sorted(selected_indices)]
            dup_method = dup_method_var.get()
            tolerance = tolerance_var.get() if dup_method == "tolerance" else None
            
            merge_window.destroy()
            self._perform_merge(selected, dup_method, tolerance)
        
        ttk.Button(button_panel, text="Merge Selected", 
                  command=confirm_merge).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_panel, text="Cancel", 
                  command=merge_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Selection counter
        def update_count():
            count_label.config(text=f"Selected: {len(selected_indices)}")
            merge_window.after(100, update_count)
        
        count_label = ttk.Label(button_panel, text="Selected: 0", foreground='blue')
        count_label.pack(side=tk.LEFT, padx=10)
        update_count()
    
    def _perform_merge(self, selected_subgraphs, dup_method, tolerance):
        """Merge selected subgraphs into current graph."""
        print(f"Merging {len(selected_subgraphs)} subgraphs...")
        
        # Collect all nodes and edges
        merged_graph = nx.Graph()
        merged_positions = {}
        merged_colors = {}
        
        # Track which nodes are duplicates
        node_map = {}  # Maps duplicate nodes to canonical node
        
        for sg_data in selected_subgraphs:
            sg = sg_data['graph']
            pos = sg_data['positions']
            col = sg_data['colors']
            
            for node in sg.nodes():
                # Check if this is a duplicate
                canonical = None
                
                if dup_method == "index":
                    # Check by lattice index (exact match)
                    for existing in merged_graph.nodes():
                        if node == existing:
                            canonical = existing
                            break
                
                elif dup_method == "tolerance":
                    # Check by position tolerance
                    node_pos = pos[node]
                    for existing, existing_pos in merged_positions.items():
                        dist = np.linalg.norm(node_pos - existing_pos)
                        if dist < tolerance:
                            canonical = existing
                            break
                
                if canonical:
                    # Duplicate found
                    node_map[node] = canonical
                    print(f"Duplicate: {node} → {canonical}")
                else:
                    # New unique node
                    merged_graph.add_node(node)
                    merged_positions[node] = pos[node].copy()
                    merged_colors[node] = col[node]
                    node_map[node] = node
            
            # Add edges with mapping
            for edge in sg.edges():
                n1 = node_map.get(edge[0], edge[0])
                n2 = node_map.get(edge[1], edge[1])
                if n1 != n2 and not merged_graph.has_edge(n1, n2):
                    weight = sg[edge[0]][edge[1]].get('weight', 1.0)
                    merged_graph.add_edge(n1, n2, weight=weight)
        
        print(f"Merged graph: {len(merged_graph.nodes())} nodes, {len(merged_graph.edges())} edges")
        
        # Now check for potential new bonds from original graphs
        potential_bonds = []
        bond_length = self.config.lattice.bond_length
        
        for sg_data in selected_subgraphs:
            if sg_data['original_graph'] is None:
                continue
            
            orig = sg_data['original_graph']
            
            # Check if any edges from original graph should be restored
            for edge in orig.edges():
                n1_mapped = node_map.get(edge[0], edge[0])
                n2_mapped = node_map.get(edge[1], edge[1])
                
                # Both nodes exist in merged graph and not already connected?
                if (merged_graph.has_node(n1_mapped) and 
                    merged_graph.has_node(n2_mapped) and 
                    not merged_graph.has_edge(n1_mapped, n2_mapped)):
                    
                    # Check distance
                    dist = np.linalg.norm(merged_positions[n1_mapped] - 
                                        merged_positions[n2_mapped])
                    if dist <= bond_length * 1.1:  # 10% tolerance
                        weight = orig[edge[0]][edge[1]].get('weight', 1.0)
                        potential_bonds.append((n1_mapped, n2_mapped, weight, dist))
        
        print(f"Found {len(potential_bonds)} potential bonds")
        
        # Show potential bonds to user
        if potential_bonds:
            self._show_potential_bonds_dialog(merged_graph, merged_positions, merged_colors, 
                                             potential_bonds)
        else:
            # No potential bonds, finalize merge
            self._finalize_merge(merged_graph, merged_positions, merged_colors)
    
    def _show_potential_bonds_dialog(self, merged_graph, merged_positions, merged_colors, potential_bonds):
        """Show dialog for user to select which potential bonds to add."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Potential Bonds Found")
        dialog.geometry("600x500")
        
        ttk.Label(dialog, 
                 text=f"Found {len(potential_bonds)} potential bonds from original graphs",
                 font=('Arial', 11, 'bold')).pack(pady=10)
        
        ttk.Label(dialog,
                 text="Select bonds to add (uncheck to exclude):",
                 font=('Arial', 9)).pack(pady=5)
        
        # Scrollable list
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner_frame = ttk.Frame(canvas)
        
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bond selection checkboxes
        bond_vars = []
        for i, (n1, n2, weight, dist) in enumerate(potential_bonds):
            var = tk.BooleanVar(value=True)
            bond_vars.append(var)
            
            cb = ttk.Checkbutton(inner_frame, 
                               text=f"Bond {i+1}: {n1} ↔ {n2}  (dist={dist:.3f} Å, weight={weight:.2f})",
                               variable=var)
            cb.pack(anchor=tk.W, pady=2)
        
        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(side=tk.BOTTOM, pady=10)
        
        def confirm():
            # Add selected bonds
            for i, var in enumerate(bond_vars):
                if var.get():
                    n1, n2, weight, _ = potential_bonds[i]
                    merged_graph.add_edge(n1, n2, weight=weight)
            
            dialog.destroy()
            self._finalize_merge(merged_graph, merged_positions, merged_colors)
        
        def cancel():
            # Cancel bond selection but still finalize merge (without the bonds)
            dialog.destroy()
            self._finalize_merge(merged_graph, merged_positions, merged_colors)
        
        def select_all():
            for var in bond_vars:
                var.set(True)
        
        def deselect_all():
            for var in bond_vars:
                var.set(False)
        
        ttk.Button(button_frame, text="Select All", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Deselect All", command=deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Confirm", command=confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
    
    def _finalize_merge(self, merged_graph, merged_positions, merged_colors):
        """Finalize the merge and update current graph."""
        self.current_graph = merged_graph
        self.current_positions = merged_positions
        self.current_colors = merged_colors
        
        # CRITICAL: Update original_graph to the merged result
        # This makes the merged graph the new "base" for undo/redo
        self.original_graph = merged_graph.copy()
        self.original_positions = merged_positions.copy()
        self.original_colors = merged_colors.copy()
        
        # Clear operation history - start fresh from merged state
        self.operation_history.clear()
        self._update_history_status()
        
        # Update display
        self.draw_current_lattice()
        
        print(f"Merge complete: {len(merged_graph.nodes())} nodes, {len(merged_graph.edges())} edges")
        print("Original graph updated to merged result, history cleared")
        messagebox.showinfo("Success", 
                          f"Merged into graph with:\n"
                          f"• {len(merged_graph.nodes())} nodes\n"
                          f"• {len(merged_graph.edges())} edges\n\n"
                          f"Operation history reset to merged state")
    
    # ========== End Subgraph Management Methods ==========


    
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
        
        # Check edit mode FIRST (priority over vacancy mode)
        if hasattr(self, 'edit_mode_var') and self.edit_mode_var:
            mode = self.edit_mode_var.get()
            if mode in ['add_bond', 'remove_bond']:
                # Find and handle node click for bonds
                clicked_node = self._find_node_at_position(event.xdata, event.ydata)
                if clicked_node:
                    self._on_node_click_edit(clicked_node)
                return
            elif mode == 'add_node' and hasattr(self, 'pending_node_add'):
                # Add node at click position
                sublattice, color = self.pending_node_add
                self._add_node_at_position(event.xdata, event.ydata, sublattice, color)
                delattr(self, 'pending_node_add')
                return
            elif mode == 'remove_node':
                # Remove node (graph editing version with undo)
                clicked_node = self._find_node_at_position(event.xdata, event.ydata)
                if clicked_node:
                    self._remove_node_edit(clicked_node)
                return
        
        # Normal vacancy/add unit modes
        if self.vacancy_mode_var.get():
            self.remove_node_at(event.xdata, event.ydata)
        elif self.add_unit_mode_var.get():
            self.add_unit_at(event.xdata, event.ydata)
    
    def remove_node_at(self, x: float, y: float):
        """Remove node at clicked position (vacancy mode)."""
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
    
    def _find_node_at_position(self, x, y, tolerance=0.3):
        """Find node near clicked position."""
        for node, pos in self.current_positions.items():
            dist = np.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            if dist < tolerance:
                return node
        return None
    
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
            # CRITICAL: Update zm_epsilon from UI value before analysis
            # This ensures user changes take effect even after load/merge
            try:
                self.physics_engine.zm_epsilon = float(self.zm_epsilon_var.get())
                print(f"Zero mode threshold: {self.physics_engine.zm_epsilon:.2e}")
            except:
                print(f"Warning: Could not update zm_epsilon, using current: {self.physics_engine.zm_epsilon:.2e}")
            
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
            else:
                # No history in file - clear it
                self.operation_history.clear()
            
            # CRITICAL FIX: After loading, the current_graph should be the new base
            # User wants to work from the loaded state, not replay old operations
            # Update original to match current and clear history
            if self.current_graph is not None:
                self.original_graph = self.current_graph.copy()
                self.original_positions = self.current_positions.copy()
                self.original_colors = self.current_colors.copy()
                
                # Clear operation history - start fresh from loaded state
                self.operation_history.clear()
                print("Loaded session: current_graph set as new base, history cleared")
            
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
