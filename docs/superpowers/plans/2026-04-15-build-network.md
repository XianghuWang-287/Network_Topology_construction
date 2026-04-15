# build_network.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone script that reads a parquet file of pairwise spectrum scores, applies GNPS-style network filtering (top-K mutual + component pruning), and exports a GraphML file.

**Architecture:** Single-file script (`build_network.py`) with four functions: `load_parquet_to_graph`, `filter_top_k`, `filter_component`, `main`. Uses pandas for I/O and networkx for graph operations. No external workflow dependencies.

**Tech Stack:** Python 3, pandas, networkx, pyarrow, argparse

---

## File Structure

- **Create:** `build_network.py` — the main script (CLI + all logic)
- **Create:** `tests/test_build_network.py` — unit tests for filtering logic

---

### Task 1: Scaffold script with CLI argument parsing

**Files:**
- Create: `build_network.py`

- [ ] **Step 1: Write the test for CLI argument parsing**

Create `tests/test_build_network.py`:

```python
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
    # We test this by importing the function directly
    from build_network import make_default_output_path
    assert make_default_output_path("/data/edges.parquet") == "/data/edges_network.graphml"
    assert make_default_output_path("foo/bar.parquet") == "foo/bar_network.graphml"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_network.py -v`
Expected: FAIL — `build_network.py` does not exist

- [ ] **Step 3: Write the CLI scaffold**

Create `build_network.py`:

```python
#!/usr/bin/env python3

import argparse
import os
import sys

import networkx as nx
import pandas as pd


def make_default_output_path(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_network.graphml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a network topology from a parquet file of pairwise scores and export as GraphML."
    )
    parser.add_argument(
        "input_parquet",
        help="Path to the input parquet file (columns: spectrum_id_1, spectrum_id_2, cosine_similarity)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output GraphML file path (default: <input_dir>/<input_name>_network.graphml)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Max edges to keep per node in mutual top-K filtering (default: 10)",
    )
    parser.add_argument(
        "--max_component_size",
        type=int,
        default=100,
        help="Max nodes per connected component (default: 100)",
    )
    args = parser.parse_args(argv)

    if args.output is None:
        args.output = make_default_output_path(args.input_parquet)

    return args


def main():
    args = parse_args()
    print(f"Input:  {args.input_parquet}")
    print(f"Output: {args.output}")
    print(f"top_k: {args.top_k}, max_component_size: {args.max_component_size}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_network.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add build_network.py tests/test_build_network.py
git commit -m "feat: scaffold build_network.py with CLI argument parsing"
```

---

### Task 2: Load parquet into networkx graph

**Files:**
- Modify: `tests/test_build_network.py`
- Modify: `build_network.py`

- [ ] **Step 1: Write the test for graph loading**

Append to `tests/test_build_network.py`:

```python
import tempfile
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
```

Add `import os` at the top of the test file if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_network.py::test_load_parquet_to_graph -v`
Expected: FAIL — `load_parquet_to_graph` not defined

- [ ] **Step 3: Implement load_parquet_to_graph**

Add to `build_network.py`, before `parse_args`:

```python
def load_parquet_to_graph(parquet_path):
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} edges")

    G = nx.Graph()

    nodes = set(df["spectrum_id_1"]).union(set(df["spectrum_id_2"]))
    G.add_nodes_from(nodes)
    print(f"  {len(nodes)} nodes")

    edges = list(zip(
        df["spectrum_id_1"],
        df["spectrum_id_2"],
        [{"score": float(s)} for s in df["cosine_similarity"]],
    ))
    G.add_edges_from(edges)
    print(f"  {len(G.edges())} edges in graph")

    return G
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_network.py::test_load_parquet_to_graph -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add build_network.py tests/test_build_network.py
git commit -m "feat: add load_parquet_to_graph function"
```

---

### Task 3: Implement top-K mutual filtering

**Files:**
- Modify: `tests/test_build_network.py`
- Modify: `build_network.py`

- [ ] **Step 1: Write the test for top-K filtering**

Append to `tests/test_build_network.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_network.py::test_filter_top_k_basic tests/test_build_network.py::test_filter_top_k_keeps_all_when_k_is_large -v`
Expected: FAIL — `filter_top_k` not defined

- [ ] **Step 3: Implement filter_top_k**

Add to `build_network.py`, after `load_parquet_to_graph`:

```python
def filter_top_k(G, top_k):
    print(f"Top-K filtering (k={top_k})...")
    print(f"  Edges before: {len(G.edges())}")

    node_cutoff_score = {}
    for node in G.nodes():
        node_edges = list(G.edges(node, data=True))
        node_edges.sort(key=lambda e: e[2]["score"], reverse=True)

        edges_to_keep = node_edges[:top_k]
        if len(edges_to_keep) == 0:
            continue
        node_cutoff_score[node] = edges_to_keep[-1][2]["score"]

    edges_to_remove = []
    for u, v, data in G.edges(data=True):
        score = data["score"]
        if u in node_cutoff_score and score < node_cutoff_score[u]:
            edges_to_remove.append((u, v))
        elif v in node_cutoff_score and score < node_cutoff_score[v]:
            edges_to_remove.append((u, v))

    for u, v in edges_to_remove:
        G.remove_edge(u, v)

    print(f"  Edges after: {len(G.edges())}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_network.py -k "filter_top_k" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add build_network.py tests/test_build_network.py
git commit -m "feat: add top-K mutual filtering"
```

---

### Task 4: Implement component size filtering

**Files:**
- Modify: `tests/test_build_network.py`
- Modify: `build_network.py`

- [ ] **Step 1: Write the test for component filtering**

Append to `tests/test_build_network.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_network.py -k "filter_component" -v`
Expected: FAIL — `filter_component` not defined

- [ ] **Step 3: Implement filter_component**

Add to `build_network.py`, after `filter_top_k`:

```python
def _prune_component(G, component, cosine_delta=0.02):
    component_edges = []
    seen = set()
    for node in component:
        for u, v, data in G.edges(node, data=True):
            edge_key = (min(u, v), max(u, v))
            if edge_key not in seen:
                seen.add(edge_key)
                component_edges.append((u, v, data))

    if not component_edges:
        return

    min_score = min(e[2]["score"] for e in component_edges)
    threshold = min_score + cosine_delta

    for u, v, data in component_edges:
        if data["score"] < threshold:
            G.remove_edge(u, v)


def filter_component(G, max_component_size):
    if max_component_size == 0:
        return

    print(f"Component filtering (max_size={max_component_size})...")

    while True:
        big_found = False
        for component in nx.connected_components(G):
            if len(component) > max_component_size:
                _prune_component(G, component)
                big_found = True
        print(f"  Edges after pruning round: {len(G.edges())}")
        if not big_found:
            break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_network.py -k "filter_component" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add build_network.py tests/test_build_network.py
git commit -m "feat: add component size filtering"
```

---

### Task 5: Label components and export GraphML

**Files:**
- Modify: `tests/test_build_network.py`
- Modify: `build_network.py`

- [ ] **Step 1: Write the test for component labeling and export**

Append to `tests/test_build_network.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_network.py -k "label_components or export_graphml" -v`
Expected: FAIL — `label_components` not defined

- [ ] **Step 3: Implement label_components**

Add to `build_network.py`, after `filter_component`:

```python
def label_components(G):
    for comp_idx, component in enumerate(nx.connected_components(G), start=1):
        for node in component:
            G.nodes[node]["component"] = comp_idx
        seen = set()
        for node in component:
            for u, v, data in G.edges(node, data=True):
                edge_key = (min(u, v), max(u, v))
                if edge_key not in seen:
                    seen.add(edge_key)
                    data["component"] = comp_idx

    print(f"  {comp_idx} components labeled")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_network.py -k "label_components or export_graphml" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add build_network.py tests/test_build_network.py
git commit -m "feat: add component labeling and GraphML export"
```

---

### Task 6: Wire up main() end-to-end

**Files:**
- Modify: `tests/test_build_network.py`
- Modify: `build_network.py`

- [ ] **Step 1: Write the end-to-end test**

Append to `tests/test_build_network.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_network.py::test_end_to_end -v`
Expected: FAIL — main() doesn't run the pipeline yet

- [ ] **Step 3: Wire up main()**

Replace the `main()` function in `build_network.py`:

```python
def main():
    args = parse_args()
    print(f"Input:  {args.input_parquet}")
    print(f"Output: {args.output}")
    print(f"top_k: {args.top_k}, max_component_size: {args.max_component_size}")
    print()

    G = load_parquet_to_graph(args.input_parquet)
    print()

    filter_top_k(G, args.top_k)
    print()

    filter_component(G, args.max_component_size)
    print()

    # Remove isolated nodes (nodes with no edges after filtering)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    if isolates:
        print(f"Removed {len(isolates)} isolated nodes")

    label_components(G)
    print()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    nx.write_graphml(G, args.output, infer_numeric_types=True)
    print(f"GraphML written to {args.output}")
    print(f"  {len(G.nodes())} nodes, {len(G.edges())} edges")
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_build_network.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add build_network.py tests/test_build_network.py
git commit -m "feat: wire up main() for end-to-end pipeline"
```

---

### Task 7: Smoke test with real data

**Files:** None (manual verification)

- [ ] **Step 1: Run on the actual parquet file**

Run:
```bash
python build_network.py data/edges_large_test_cpu.parquet
```

Expected output: progress messages showing edge/node counts at each filtering stage, and a `data/edges_large_test_cpu_network.graphml` file created.

- [ ] **Step 2: Verify the output file**

Run:
```bash
python -c "
import networkx as nx
G = nx.read_graphml('data/edges_large_test_cpu_network.graphml')
print(f'Nodes: {len(G.nodes())}')
print(f'Edges: {len(G.edges())}')
components = list(nx.connected_components(G))
print(f'Components: {len(components)}')
sizes = [len(c) for c in components]
print(f'Largest component: {max(sizes)} nodes')
print(f'Smallest component: {min(sizes)} nodes')
"
```

Expected: largest component <= 100 nodes.

- [ ] **Step 3: Commit (if any fixes were needed)**

```bash
git add build_network.py
git commit -m "fix: adjustments from smoke test with real data"
```
