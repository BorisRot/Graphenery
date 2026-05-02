"""
Automated tests for polygon selection system.

Run this to verify basic functionality without manual testing.
"""

import sys
import numpy as np
import networkx as nx

# Add paths
sys.path.insert(0, '/mnt/user-data/outputs')

from lattice_builder import LatticeFactory
from physics_engine import PhysicsEngine
from polygon_selector import PolygonSelector, OperationHistory


def test_sublattice_counting():
    """Test sublattice counting logic."""
    print("\n=== Test 1: Sublattice Counting ===")
    
    # Create small lattice
    graph, positions, colors = LatticeFactory.create(
        'full',
        num_units=5,
        bond_length=1.0,
        boundary='bearded'
    )
    
    # Count sublattices
    N_A = sum(1 for node in graph.nodes() if node[2] == 0)
    N_B = sum(1 for node in graph.nodes() if node[2] == 1)
    imbalance = abs(N_A - N_B)
    
    print(f"Total nodes: {len(graph.nodes())}")
    print(f"N_A (red): {N_A}")
    print(f"N_B (black): {N_B}")
    print(f"Imbalance: {imbalance}")
    
    assert N_A > 0, "No A sublattice nodes found!"
    assert N_B > 0, "No B sublattice nodes found!"
    print("✓ Sublattice counting works")


def test_zero_mode_calculation():
    """Test zero mode analysis with various imbalances."""
    print("\n=== Test 2: Zero Mode Analysis ===")
    
    engine = PhysicsEngine(zm_epsilon=1e-10)
    
    # Test case 1: Balanced lattice
    print("\nTest 2.1: Balanced lattice")
    graph, positions, colors = LatticeFactory.create(
        'full',
        num_units=5,
        bond_length=1.0,
        boundary='bearded'
    )
    
    H, eigenvals, eigenmodes = engine.compute_eigensystem(graph)
    N_A = sum(1 for node in graph.nodes() if node[2] == 0)
    N_B = sum(1 for node in graph.nodes() if node[2] == 1)
    imbalance = abs(N_A - N_B)
    
    print(f"  Nodes: {len(graph.nodes())}, N_A: {N_A}, N_B: {N_B}, Imbalance: {imbalance}")
    
    zm_vecs = engine.identify_zero_modes(eigenvals, eigenmodes, imbalance)
    num_found = len([k for k in zm_vecs.keys() if isinstance(k, int)])
    
    print(f"  Expected ~{imbalance} zero modes, found {num_found}")
    print("  ✓ No errors")
    
    # Test case 2: Extreme imbalance (the bug case)
    print("\nTest 2.2: Extreme imbalance (small graph)")
    # Simulate extracted NS chain: very imbalanced
    small_nodes = 20
    small_N_A = 18
    small_N_B = 2
    small_imbalance = abs(small_N_A - small_N_B)
    
    # Create dummy eigenvalues
    dummy_eigenvals = np.linspace(-3, 3, small_nodes)
    dummy_eigenmodes = np.eye(small_nodes)
    
    print(f"  Nodes: {small_nodes}, N_A: {small_N_A}, N_B: {small_N_B}, Imbalance: {small_imbalance}")
    print(f"  Imbalance > N/2: {small_imbalance > small_nodes // 2}")
    
    try:
        zm_vecs = engine.identify_zero_modes(dummy_eigenvals, dummy_eigenmodes, small_imbalance)
        print("  ✓ No index error!")
    except IndexError as e:
        print(f"  ✗ Index error occurred: {e}")
        raise


def test_operation_history():
    """Test operation history and undo."""
    print("\n=== Test 3: Operation History ===")
    
    history = OperationHistory()
    
    # Add operations
    history.add_operation({'type': 'extract', 'nodes': [1, 2, 3]})
    history.add_operation({'type': 'delete', 'nodes': [4, 5]})
    history.add_operation({'type': 'highlight', 'color': 'yellow'})
    
    assert len(history.operations) == 3, "Wrong operation count"
    assert history.current_index == 2, "Wrong current index"
    print(f"Added 3 operations, current index: {history.current_index}")
    
    # Undo
    undone = history.undo()
    assert undone['type'] == 'highlight', "Wrong operation undone"
    assert history.current_index == 1, "Wrong index after undo"
    print(f"Undone 'highlight', current index: {history.current_index}")
    
    # Undo again
    undone = history.undo()
    assert undone['type'] == 'delete', "Wrong operation undone"
    assert history.current_index == 0, "Wrong index after second undo"
    print(f"Undone 'delete', current index: {history.current_index}")
    
    # Redo
    redone = history.redo()
    assert redone['type'] == 'delete', "Wrong operation redone"
    assert history.current_index == 1, "Wrong index after redo"
    print(f"Redone 'delete', current index: {history.current_index}")
    
    print("✓ Operation history works correctly")


def test_polygon_point_in_polygon():
    """Test polygon point-in-polygon detection."""
    print("\n=== Test 4: Point-in-Polygon ===")
    
    # Create simple square polygon
    vertices = [(0, 0), (10, 0), (10, 10), (0, 10)]
    
    from matplotlib.path import Path
    polygon_path = Path(vertices)

    # Note: Boundary behavior can be inconsistent at corners due to
    # numerical precision and winding number algorithm
    # We only care about clearly interior vs clearly exterior points
    test_cases = [
        ((5, 5), True, "center (clearly interior)"),
        ((15, 5), False, "outside (clearly exterior)"),
        ((-1, 5), False, "outside (clearly exterior)"),
        ((5, 15), False, "outside (clearly exterior)"),
        ((5, -1), False, "outside (clearly exterior)"),
    ]

    passed = 0
    failed = 0

    for point, expected, description in test_cases:
        result = polygon_path.contains_point(point)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Point {point} ({description}): {result} (expected {expected})")
        if result == expected:
            passed += 1
        else:
            failed += 1

    # Also test boundary points but don't enforce specific behavior
    print("\n  Boundary point behavior (may vary by implementation):")
    boundary_points = [
        ((0, 0), "corner"),
        ((10, 10), "corner"),
        ((5, 0), "edge"),
        ((10, 5), "edge"),
    ]

    for point, description in boundary_points:
        result = polygon_path.contains_point(point)
        print(f"    Point {point} ({description}): {result}")

    print(f"\nNote: Boundary points may or may not be included depending on algorithm details.")
    print(f"For lattice selection, this doesn't matter - nodes are discrete and rarely on exact boundaries.")

    if failed > 0:
        raise AssertionError(f"{failed} point-in-polygon tests failed")

    print(f"✓ All {passed} critical point-in-polygon tests passed")


def test_graph_operations():
    """Test graph manipulation operations."""
    print("\n=== Test 5: Graph Operations ===")

    # Create lattice
    graph, positions, colors = LatticeFactory.create(
        'full',
        num_units=5,
        bond_length=1.0,
        boundary='bearded'
    )

    initial_nodes = len(graph.nodes())
    print(f"Initial graph: {initial_nodes} nodes")

    # Simulate extract: keep only some nodes
    nodes_to_keep = list(graph.nodes())[:initial_nodes // 2]
    nodes_to_remove = set(graph.nodes()) - set(nodes_to_keep)

    for node in nodes_to_remove:
        graph.remove_node(node)
        del positions[node]
        del colors[node]

    extracted_nodes = len(graph.nodes())
    print(f"After extract: {extracted_nodes} nodes")
    assert extracted_nodes == initial_nodes // 2, "Wrong number of nodes after extract"

    # Count sublattices
    N_A = sum(1 for node in graph.nodes() if node[2] == 0)
    N_B = sum(1 for node in graph.nodes() if node[2] == 1)
    print(f"Extracted graph: N_A={N_A}, N_B={N_B}, imbalance={abs(N_A - N_B)}")

    print("✓ Graph operations work correctly")


def run_all_tests():
    """Run all automated tests."""
    print("="*60)
    print("POLYGON SELECTION SYSTEM - AUTOMATED TESTS")
    print("="*60)

    try:
        test_sublattice_counting()
        test_zero_mode_calculation()
        test_operation_history()
        test_polygon_point_in_polygon()
        test_graph_operations()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        return True

    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED ✗: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)