import subprocess
import sys

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "build_network.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "input_parquet" in result.stdout
    assert "--output" in result.stdout
    assert "--top_k" in result.stdout
    assert "--max_component_size" in result.stdout

def test_cli_default_output_path():
    """Verify default output path is derived from input path."""
    from build_network import make_default_output_path
    assert make_default_output_path("/data/edges.parquet") == "/data/edges_network.graphml"
    assert make_default_output_path("foo/bar.parquet") == "foo/bar_network.graphml"


import tempfile
import os
import pandas as pd
import networkx as nx

def test_load_parquet_to_graph():
    df = pd.DataFrame({
        "spectrum_id_1": ["A", "A", "B"],
        "spectrum_id_2": ["B", "C", "C"],
        "cosine_similarity": [0.9, 0.8, 0.7],
    })
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        from build_network import load_parquet_to_graph
        G = load_parquet_to_graph(f.name)

    assert isinstance(G, nx.Graph)
    assert set(G.nodes()) == {"A", "B", "C"}
    assert len(G.edges()) == 3
    assert G["A"]["B"]["score"] == 0.9
    assert G["A"]["C"]["score"] == 0.8
    assert G["B"]["C"]["score"] == 0.7
    os.unlink(f.name)


def test_filter_top_k_basic():
    """With top_k=1, only the best edge per node survives, and only if mutual."""
    from build_network import filter_top_k

    G = nx.Graph()
    # Node A connects to B (0.9) and C (0.8)
    # Node B connects to A (0.9) and C (0.7)
    # Node C connects to A (0.8) and B (0.7)
    G.add_edge("A", "B", score=0.9)
    G.add_edge("A", "C", score=0.8)
    G.add_edge("B", "C", score=0.7)

    filter_top_k(G, top_k=1)

    # A's top-1 is B (0.9). B's top-1 is A (0.9). Mutual -> kept.
    # A-C: A's top-1 cutoff is 0.9, C's top-1 cutoff is 0.8. 0.8 < 0.9 -> removed.
    # B-C: B's top-1 cutoff is 0.9, C's top-1 cutoff is 0.8. 0.7 < both -> removed.
    assert set(G.edges()) == {("A", "B")}


def test_filter_top_k_keeps_all_when_k_is_large():
    """When top_k >= number of edges per node, nothing is removed."""
    from build_network import filter_top_k

    G = nx.Graph()
    G.add_edge("A", "B", score=0.9)
    G.add_edge("A", "C", score=0.8)
    G.add_edge("B", "C", score=0.7)

    filter_top_k(G, top_k=10)

    assert len(G.edges()) == 3


def test_filter_component_breaks_large_component():
    """A component of 4 nodes should be pruned when max_component_size=3."""
    from build_network import filter_component

    G = nx.Graph()
    # Chain: A-B-C-D, all connected linearly
    G.add_edge("A", "B", score=0.9)
    G.add_edge("B", "C", score=0.7)  # weakest
    G.add_edge("C", "D", score=0.8)

    filter_component(G, max_component_size=3)

    # Component has 4 nodes > 3. Min score = 0.7.
    # Prune threshold = 0.7 + 0.02 = 0.72.
    # Edge B-C (0.7) < 0.72 -> removed.
    # Now two components: {A,B} and {C,D}, both <= 3.
    assert not G.has_edge("B", "C")
    assert G.has_edge("A", "B")
    assert G.has_edge("C", "D")


def test_filter_component_no_change_when_small():
    """Components already within limit should not be changed."""
    from build_network import filter_component

    G = nx.Graph()
    G.add_edge("A", "B", score=0.9)
    G.add_edge("C", "D", score=0.8)

    filter_component(G, max_component_size=100)

    assert len(G.edges()) == 2


def test_label_components():
    from build_network import label_components

    G = nx.Graph()
    G.add_edge("A", "B", score=0.9)
    G.add_edge("C", "D", score=0.8)

    label_components(G)

    # Two components, each gets a different index
    assert G.nodes["A"]["component"] == G.nodes["B"]["component"]
    assert G.nodes["C"]["component"] == G.nodes["D"]["component"]
    assert G.nodes["A"]["component"] != G.nodes["C"]["component"]

    # Edges get component too
    assert G["A"]["B"]["component"] == G.nodes["A"]["component"]
    assert G["C"]["D"]["component"] == G.nodes["C"]["component"]


def test_export_graphml_roundtrip():
    from build_network import label_components

    G = nx.Graph()
    G.add_edge("A", "B", score=0.9)
    G.add_edge("C", "D", score=0.8)
    label_components(G)

    with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
        nx.write_graphml(G, f.name, infer_numeric_types=True)
        G2 = nx.read_graphml(f.name)

    assert set(G2.nodes()) == {"A", "B", "C", "D"}
    assert len(G2.edges()) == 2
    assert float(G2["A"]["B"]["score"]) == 0.9
    os.unlink(f.name)


def test_end_to_end():
    """Full pipeline: parquet -> filter -> graphml."""
    df = pd.DataFrame({
        "spectrum_id_1": ["A", "A", "A", "B", "B", "C"],
        "spectrum_id_2": ["B", "C", "D", "C", "D", "D"],
        "cosine_similarity": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
    })
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as pf:
        df.to_parquet(pf.name)
        parquet_path = pf.name

    output_path = parquet_path.replace(".parquet", "_network.graphml")

    result = subprocess.run(
        [sys.executable, "build_network.py", parquet_path, "-o", output_path, "--top_k", "2", "--max_component_size", "100"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    G = nx.read_graphml(output_path)
    assert len(G.nodes()) > 0
    assert len(G.edges()) > 0

    # Check that component attribute exists on nodes
    for node in G.nodes():
        assert "component" in G.nodes[node]

    # Check that score attribute exists on edges
    for u, v in G.edges():
        assert "score" in G[u][v]

    os.unlink(parquet_path)
    os.unlink(output_path)
