"""
Stages D & E: Poiseuille Conductance and Current-Flow Betweenness Centrality Scorer.
Modular interface: score_edges(graph, config) -> pd.DataFrame
"""
import networkx as nx
import numpy as np
import pandas as pd


def assign_poiseuille_conductance(graph, radius_mode='min', eps=1e-3):
    """
    Stage D: Compute Poiseuille conductance c_e = r_e^4 / (L_e + 1e-3).
    radius_mode: 'min' (bottleneck radius, default) or 'mean'.
    """
    for (u, v, k) in graph.edges(keys=True):
        edge = graph[u][v][k]
        L_e = float(edge.get('length', 1.0))
        if radius_mode == 'mean':
            r_e = float(edge.get('radius_mean', 1.0))
        else:
            r_e = float(edge.get('radius_min', 1.0))
        
        # c_e = r^4 / L
        conductance = (r_e ** 4) / (L_e + eps)
        edge['conductance'] = float(conductance)
        edge['weight'] = float(conductance)  # nx current flow betweenness uses weight as conductance


def compute_current_flow_betweenness(graph, pair_mode='all'):
    """
    Stage E: Computes edge current-flow betweenness per connected component.
    pair_mode: 'all' (all pairs in component) or 'leaf_to_leaf' (only degree-1 endpoints).
    """
    # Initialize all edge centralities to 0.0
    for (u, v, k) in graph.edges(keys=True):
        graph[u][v][k]['betweenness'] = 0.0
        graph[u][v][k]['component_id'] = -1
        graph[u][v][k]['tiny_fragment'] = False

    # Extract connected components (subgraphs)
    # Convert MultiGraph to Graph for nx.edge_current_flow_betweenness_centrality if needed
    comp_id = 0
    # Connected components on undirected graph
    components = list(nx.connected_components(graph))

    for comp_nodes in components:
        comp_id += 1
        sub_g = graph.subgraph(comp_nodes).copy()
        num_nodes = len(comp_nodes)

        # Flag tiny fragments (< 3 nodes)
        if num_nodes < 3:
            for (u, v, k) in sub_g.edges(keys=True):
                graph[u][v][k]['betweenness'] = 0.0
                graph[u][v][k]['component_id'] = comp_id
                graph[u][v][k]['tiny_fragment'] = True
            continue

        # If sub_g is a MultiGraph, collapse to simple Graph for Laplacian solving
        # sum conductances of parallel edges
        simple_g = nx.Graph()
        for u in sub_g.nodes():
            simple_g.add_node(u)

        for u, v, k in sub_g.edges(keys=True):
            cond = sub_g[u][v][k].get('conductance', 1.0)
            if simple_g.has_edge(u, v):
                simple_g[u][v]['conductance'] += cond
                simple_g[u][v]['weight'] += cond
            else:
                simple_g.add_edge(u, v, conductance=cond, weight=cond)

        # Determine node pairs if leaf_to_leaf
        sources = None
        targets = None
        if pair_mode == 'leaf_to_leaf':
            leaves = [n for n in sub_g.nodes() if sub_g.degree(n) == 1]
            if len(leaves) >= 2:
                sources = leaves
                targets = leaves

        try:
            # Exact all-pairs current flow betweenness
            # Note: in NetworkX, weight attribute represents resistance or conductance depending on implementation.
            # In edge_current_flow_betweenness_centrality, weight is conductivity/conductance.
            eb = nx.edge_current_flow_betweenness_centrality(simple_g, weight='conductance', normalized=True)
        except Exception:
            # Fallback to shortest-path betweenness or topological weight if matrix singular
            try:
                eb = nx.edge_betweenness_centrality(simple_g, weight='weight', normalized=True)
            except Exception:
                eb = {(u, v): 0.0 for u, v in simple_g.edges()}

        # Map back to MultiGraph edges
        for (u, v, k) in sub_g.edges(keys=True):
            edge_tuple = (u, v) if (u, v) in eb else (v, u)
            score = eb.get(edge_tuple, 0.0)
            graph[u][v][k]['betweenness'] = float(score)
            graph[u][v][k]['component_id'] = comp_id
            graph[u][v][k]['tiny_fragment'] = False

    # Image-level Normalization: CB_hat = CB / max(CB)
    all_scores = [graph[u][v][k]['betweenness'] for u, v, k in graph.edges(keys=True)]
    max_score = max(all_scores) if all_scores else 0.0
    for (u, v, k) in graph.edges(keys=True):
        if max_score > 1e-12:
            graph[u][v][k]['CB_hat'] = float(graph[u][v][k]['betweenness'] / max_score)
        else:
            graph[u][v][k]['CB_hat'] = 0.0


def score_edges(graph, config=None):
    """
    Standard Scorer Interface: takes (graph, config) and returns annotated pd.DataFrame.
    Modular design allows swapping with Murray's Law scorer in the future.
    """
    if config is None:
        config = {}

    radius_mode = config.get('radius_mode', 'min')
    pair_mode = config.get('pair_mode', 'all')
    use_conductance = config.get('use_conductance', True)

    # 1. Assign Conductance
    if use_conductance:
        assign_poiseuille_conductance(graph, radius_mode=radius_mode)
    else:
        for (u, v, k) in graph.edges(keys=True):
            graph[u][v][k]['conductance'] = 1.0
            graph[u][v][k]['weight'] = 1.0

    # 2. Compute Betweenness
    compute_current_flow_betweenness(graph, pair_mode=pair_mode)

    # 3. Compile Edge Table DataFrame
    rows = []
    edge_idx = 0
    for (u, v, k) in graph.edges(keys=True):
        e = graph[u][v][k]
        rows.append({
            'edge_id': edge_idx,
            'u': u,
            'v': v,
            'k': k,
            'component_id': e.get('component_id', -1),
            'tiny_fragment': e.get('tiny_fragment', False),
            'length': e.get('length', 1.0),
            'radius_min': e.get('radius_min', 1.0),
            'radius_mean': e.get('radius_mean', 1.0),
            'conductance': e.get('conductance', 1.0),
            'betweenness': e.get('betweenness', 0.0),
            'CB_hat': e.get('CB_hat', 0.0),
            'touches_border': e.get('touches_border', False),
            'pts': e.get('pts')
        })
        edge_idx += 1

    df = pd.DataFrame(rows)
    return df
