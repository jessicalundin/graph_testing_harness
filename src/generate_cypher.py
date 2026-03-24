import networkx as nx
import os
from pathlib import Path
from generate_graph import load_and_build_graph, create_node_type_map

# Paths - Mac-compatible using pathlib
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
CYPHER_PATH = PROJECT_ROOT / 'databases' / 'IMCI_schema.cypherl'
GRAPH_PATH = PROJECT_ROOT / 'databases' / 'IMCI_schema.graphml'

def generate_cypher_from_graph(graph, node_type_map):
    """Generate Cypher statements from the NetworkX graph."""
    cypher_statements = set()
    
    # Generate CREATE statements for all nodes
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        if node_type == 'Condition':
            age_range = data.get('age_range', '2-60')
            cypher_statements.add(f"CREATE (c:Condition {{name:'{node}', age_range:'{age_range}'}})")
        elif node_type == 'Symptom':
            cypher_statements.add(f"CREATE (s:Symptom {{name:'{node}'}})")
        elif node_type == 'Treatment':
            cypher_statements.add(f"CREATE (t:Treatment {{name:'{node}'}})")
        elif node_type == 'FollowUp':
            cypher_statements.add(f"CREATE (f:FollowUp {{name:'{node}'}})")
        elif node_type == 'Severity':
            cypher_statements.add(f"CREATE (sev:Severity {{name:'{node}'}})")
    
    # Generate MATCH and CREATE statements for relationships
    for source, target, data in graph.edges(data=True):
        edge_type = data.get('type', 'CONNECTS')
        
        if edge_type == 'INDICATES':  # Symptom -> Condition
            cypher_statements.add(f"MATCH (s:Symptom {{name:'{source}'}}), (c:Condition {{name:'{target}'}}) CREATE (s)-[:INDICATES]->(c)")
        elif edge_type == 'TREAT':  # Condition -> Treatment
            cypher_statements.add(f"MATCH (c:Condition {{name:'{source}'}}), (t:Treatment {{name:'{target}'}}) CREATE (c)-[:TREAT]->(t)")
        elif edge_type == 'FOLLOW':  # Condition -> FollowUp
            cypher_statements.add(f"MATCH (c:Condition {{name:'{source}'}}), (f:FollowUp {{name:'{target}'}}) CREATE (c)-[:FOLLOW]->(f)")
        elif edge_type == 'TRIAGE':  # Condition -> Severity
            cypher_statements.add(f"MATCH (c:Condition {{name:'{source}'}}), (sev:Severity {{name:'{target}'}}) CREATE (c)-[:TRIAGE]->(sev)")
    
    return cypher_statements

def save_cypher_schema(cypher_statements):
    """Save Cypher statements to file."""
    os.makedirs(os.path.dirname(CYPHER_PATH), exist_ok=True)
    with open(CYPHER_PATH, 'w') as f:
        for stmt in sorted(cypher_statements):
            f.write(stmt + '\n')
    print(f"Cypher schema saved to {CYPHER_PATH}")

def load_existing_graph():
    """Load existing graph from GraphML file if it exists, otherwise build new one."""
    if os.path.exists(GRAPH_PATH):
        print(f"Loading existing graph from {GRAPH_PATH}")
        return nx.read_graphml(GRAPH_PATH)
    else:
        print("No existing graph found, building new one...")
        return load_and_build_graph()

def main():
    """Main function to generate Cypher schema."""
    print("Loading graph...")
    graph = load_existing_graph()
    
    print("Creating node type mapping...")
    node_type_map = create_node_type_map(graph)
    
    print("Generating Cypher statements...")
    cypher_statements = generate_cypher_from_graph(graph, node_type_map)
    
    print("Saving Cypher schema...")
    save_cypher_schema(cypher_statements)
    
    print(f"Generated {len(cypher_statements)} Cypher statements.")
    
    # Print summary
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    print(f"Node types: {list(node_type_map.keys())}")
    for node_type, nodes in node_type_map.items():
        print(f"  {node_type}: {len(nodes)} nodes")

if __name__ == "__main__":
    main()