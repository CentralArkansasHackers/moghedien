# tests/test_bloodhound_data.py
import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))

from moghedien.bloodhound.parser import BloodHoundParser
from moghedien.core.graph import ADGraph
from moghedien.core.pathfinder import AttackPathFinder

def test_bloodhound_parser():
    """Test parsing BloodHound data."""
    parser = BloodHoundParser()
    
    # Path to the test data directory
    data_dir = "test_data"
    
    # Parse all JSON files in the directory
    parser.parse_directory(data_dir)
    
    # Verify that we parsed some data
    assert len(parser.domains) > 0, "No domains parsed"
    assert len(parser.computers) > 0, "No computers parsed"
    assert len(parser.users) > 0, "No users parsed"
    assert len(parser.groups) > 0, "No groups parsed"
    
    print(f"Successfully parsed {len(parser.domains)} domains")
    print(f"Successfully parsed {len(parser.computers)} computers")
    print(f"Successfully parsed {len(parser.users)} users")
    print(f"Successfully parsed {len(parser.groups)} groups")
    print(f"Successfully parsed {len(parser.ous)} OUs")
    print(f"Successfully parsed {len(parser.gpos)} GPOs")
    print(f"Successfully parsed {len(parser.containers)} containers")
    
    return parser

def test_graph_builder(parser):
    """Test building the graph from parsed data."""
    graph = ADGraph()
    graph.build_from_bloodhound(parser)
    
    # Verify that the graph has nodes and edges
    node_count = graph.graph.number_of_nodes()
    edge_count = graph.graph.number_of_edges()
    
    assert node_count > 0, "Graph has no nodes"
    assert edge_count > 0, "Graph has no edges"
    
    print(f"Graph built with {node_count} nodes and {edge_count} edges")
    
    return graph

def test_pathfinder(graph):
    """Test the attack path finder."""
    pathfinder = AttackPathFinder(graph)
    
    # Find a user node to use as the source
    user_node = None
    for node_id, attrs in graph.graph.nodes(data=True):
        if attrs.get('type') == 'User':
            user_node = node_id
            break
    
    if not user_node:
        print("No user node found")
        return
        
    # Find paths to Domain Admin
    print(f"Finding paths from {graph.graph.nodes[user_node].get('name', user_node)} to Domain Admin")
    paths = pathfinder.find_paths_to_domain_admin(user_node)
    
    if paths:
        print(f"Found {len(paths)} paths to Domain Admin")
        print("Top path:")
        print(f"  Score: {paths[0]['total_score']:.2f}")
        print(f"  Success: {paths[0]['success_score']:.2f}%")
        print(f"  Stealth: {paths[0]['stealth_score']:.2f}%")
        print(f"  Length: {paths[0]['length']} steps")
        
        for i, step in enumerate(paths[0]['steps']):
            print(f"  Step {i+1}: {step['source']} -> {step['relationship']} -> {step['target']}")
    else:
        print("No paths found to Domain Admin")
    
    # Find paths to high-value targets
    print(f"\nFinding paths from {graph.graph.nodes[user_node].get('name', user_node)} to high-value targets")
    paths = pathfinder.find_paths_to_high_value_targets(user_node)
    
    if paths:
        print(f"Found {len(paths)} paths to high-value targets")
        print("Top path:")
        print(f"  Score: {paths[0]['total_score']:.2f}")
        print(f"  Success: {paths[0]['success_score']:.2f}%")
        print(f"  Stealth: {paths[0]['stealth_score']:.2f}%")
        print(f"  Length: {paths[0]['length']} steps")
        
        for i, step in enumerate(paths[0]['steps']):
            print(f"  Step {i+1}: {step['source']} -> {step['relationship']} -> {step['target']}")
    else:
        print("No paths found to high-value targets")

if __name__ == "__main__":
    # Check if test data directory exists
    if not os.path.isdir("test_data"):
        os.makedirs("test_data", exist_ok=True)
        print("Created test_data directory. Please copy BloodHound JSON files there.")
        sys.exit(1)
        
    print("Testing BloodHound parser...")
    parser = test_bloodhound_parser()
    
    print("\nTesting graph builder...")
    graph = test_graph_builder(parser)
    
    print("\nTesting attack path finder...")
    test_pathfinder(graph)
