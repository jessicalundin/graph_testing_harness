import pandas as pd
import networkx as nx
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'memgraph_input.txt'
GRAPH_PATH = PROJECT_ROOT / 'databases' / 'IMCI_schema.graphml'

def map_severity_color_to_word(color):
    """Map severity color to descriptive word."""
    severity_map = {
        'Pink': 'severe',
        'Yellow': 'moderate', 
        'Green': 'mild'
    }
    return severity_map.get(color, color)

def load_and_build_graph():
    """Load CSV data and build the graph structure."""
    df = pd.read_csv(DATA_PATH, sep='\t', encoding='latin-1')
    graph = nx.MultiDiGraph()
    
    for _, row in df.iterrows():
        cond = row['condition']
        age_range = row['age range']
        graph.add_node(cond, type='Condition', age_range=age_range)
        
        # Add symptoms
        if isinstance(row['symptoms'], str):
            for symp in row['symptoms'].split(';'):
                symp = symp.strip()
                graph.add_node(symp, type='Symptom')
                graph.add_edge(symp, cond, type='INDICATES')
        
        # Add treatments
        if isinstance(row['treatment'], str):
            for treat in row['treatment'].split(';'):
                treat = treat.strip()
                graph.add_node(treat, type='Treatment')
                graph.add_edge(cond, treat, type='TREAT')
        
        # Add follow-ups
        if isinstance(row['follow-up'], str):
            for follow in row['follow-up'].split(';'):
                follow = follow.strip()
                graph.add_node(follow, type='FollowUp')
                graph.add_edge(cond, follow, type='FOLLOW')
        
        # Add severity
        if isinstance(row['severity'], str):
            sev = row['severity'].strip()
            # Convert color to word
            sev_word = map_severity_color_to_word(sev)
            graph.add_node(sev_word, type='Severity')
            graph.add_edge(cond, sev_word, type='TRIAGE')
    
    
    return graph

def create_node_type_map(graph):
    """Create a mapping of node types to their nodes."""
    node_type_map = {}
    for n, d in graph.nodes(data=True):
        t = d['type']
        node_type_map.setdefault(t, []).append(n)
    return node_type_map

def save_graph(graph):
    """Save NetworkX graph as GraphML."""
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    nx.write_graphml(graph, GRAPH_PATH)
    print(f"Graph saved to {GRAPH_PATH}")

def main():
    """Main function to build and save the graph."""
    print("Loading data and building graph...")
    graph = load_and_build_graph()
    
    print("Creating node type mapping...")
    node_type_map = create_node_type_map(graph)
    
    print("Saving graph...")
    save_graph(graph)
    
    print(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    print(f"Node types: {list(node_type_map.keys())}")
    for node_type, nodes in node_type_map.items():
        print(f"  {node_type}: {len(nodes)} nodes")

if __name__ == "__main__":
    main()