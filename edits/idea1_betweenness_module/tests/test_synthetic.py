"""
Unit Test: Synthetic vascular graph verifying current-flow betweenness behavior.
Graph Topology:
- Trunk: Node 1 to Node 2
- Bridge: Node 2 to Node 3 (connects left subtree to right subtree - bottleneck)
- Loop: Node 3 branches to Node 4 and Node 5, which reconverge at Node 6 (redundant loop)
- Dead-end Spur: Node 2 branches to Node 7 (leaf node / dead-end spur)

Theoretical Expectations:
1. Bridge edge (2-3): HIGHEST betweenness (all current across clusters must flow through it).
2. Dead-end spur edge (2-7): LOWEST / near-zero betweenness (no current flows through dead ends).
3. Loop edges (3-4, 3-5, 4-6, 5-6): MODERATE betweenness (current splits across parallel paths).
"""
import sys
import os
import networkx as nx
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scoring import score_edges


def build_synthetic_graph():
    G = nx.MultiGraph()

    # Define edges: (u, v, length, radius)
    edges = [
        # Trunk (1 -> 2)
        (1, 2, 10.0, 3.0),
        # Dead-end spur off node 2 (2 -> 7)
        (2, 7, 5.0, 1.0),
        # Critical Bridge (2 -> 3)
        (2, 3, 8.0, 2.5),
        # Redundant Loop (3 -> 4, 3 -> 5, 4 -> 6, 5 -> 6)
        (3, 4, 6.0, 2.0),
        (3, 5, 6.0, 2.0),
        (4, 6, 6.0, 2.0),
        (5, 6, 6.0, 2.0),
        # Terminal leaf off node 6 (6 -> 8)
        (6, 8, 8.0, 2.0)
    ]

    for u, v, L, r in edges:
        G.add_edge(u, v, length=L, radius_min=r, radius_mean=r, pts=np.array([[0, 0], [1, 1]]))

    return G


def test_synthetic_betweenness():
    print("=" * 65)
    print(" RUNNING SYNTHETIC GRAPH UNIT TEST FOR CURRENT-FLOW BETWEENNESS")
    print("=" * 65)

    G = build_synthetic_graph()
    df = score_edges(G, config={'radius_mode': 'min', 'use_conductance': True})

    print(f"\nExtracted {len(df)} edges in synthetic vascular network:")
    for _, row in df.iterrows():
        print(f"Edge ({row['u']} <-> {row['v']}): Length={row['length']:.1f}, "
              f"Radius={row['radius_min']:.1f}, Conductance={row['conductance']:.4f}, "
              f"Raw CB={row['betweenness']:.4f}, CB_hat={row['CB_hat']:.4f}")

    # Retrieve scores by edge
    scores = {}
    for _, row in df.iterrows():
        pair = tuple(sorted([row['u'], row['v']]))
        scores[pair] = row['CB_hat']

    bridge_score = scores[(2, 3)]
    spur_score = scores[(2, 7)]
    loop_scores = [scores[(3, 4)], scores[(3, 5)], scores[(4, 6)], scores[(5, 6)]]
    avg_loop_score = np.mean(loop_scores)

    print("\n" + "-" * 50)
    print(" THEORETICAL VALIDATION CHECKS:")
    print("-" * 50)
    print(f"1. Critical Bridge (2-3) Score:  {bridge_score:.4f} (Max expected = 1.00)")
    print(f"2. Dead-end Spur   (2-7) Score:  {spur_score:.4f} (Near-zero expected)")
    print(f"3. Redundant Loop  Avg Score:    {avg_loop_score:.4f} (Moderate expected)")

    # Assertions
    assert bridge_score >= 0.99, f"Expected Bridge (2-3) to be near max (1.0), got {bridge_score}"
    assert spur_score < avg_loop_score, f"Expected Spur < Loop, got Spur={spur_score}, Loop={avg_loop_score}"
    assert bridge_score > avg_loop_score, f"Expected Bridge > Loop, got Bridge={bridge_score}, Loop={avg_loop_score}"

    print("\n[SUCCESS] ALL SYNTHETIC GRAPH UNIT TESTS PASSED PERFECTLY! [OK]")
    print("=" * 65)


if __name__ == "__main__":
    test_synthetic_betweenness()
