# Design: build_network.py — Parquet to GraphML Network Topology

## Summary

A standalone Python script that reads a parquet file of pairwise spectrum scores, applies the same network filtering algorithm as the GNPS Classical Networking Workflow (top-K mutual filtering + component size pruning), and outputs a GraphML file.

## Input

- **File format:** Parquet
- **Required columns:** `spectrum_id_1`, `spectrum_id_2`, `cosine_similarity`
- **Example data:** `data/edges_large_test_cpu.parquet` (149M rows, ~898 MB)

## Output

- **File format:** GraphML
- **Default path:** `<input_dir>/<input_basename>_network.graphml`
- **Node attributes:** `component` (int, connected component index)
- **Edge attributes:** `score` (float, cosine similarity), `component` (int, connected component index)

## CLI Interface

```
python build_network.py <input_parquet> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input_parquet` | (required, positional) | Path to input parquet file |
| `--output` / `-o` | auto-generated from input path | Output GraphML file path |
| `--top_k` | `10` | Max edges per node (mutual filtering) |
| `--max_component_size` | `100` | Max nodes per connected component |

## Algorithm

Ported directly from `Classical_Networking_Workflow/GNPS_sharedcode/molecular_network_filtering_library.py`. Four steps:

### Step 1 — Load and build graph

- Read parquet with pandas (full load, ~30-50 GB memory)
- Build `nx.Graph()` with all spectrum IDs as nodes
- Add edges with `score` attribute = `cosine_similarity`

### Step 2 — Top-K mutual filtering

Identical to `molecular_network_filtering_library.filter_top_k()`:

1. For each node, sort its edges by score descending, record the score of the K-th edge as that node's cutoff
2. For each edge: if its score < either endpoint's cutoff, mark for removal
3. Remove all marked edges

Effect: an edge survives only if it is in both endpoints' top-K.

### Step 3 — Component size filtering

Identical to `molecular_network_filtering_library.filter_component()` + `prune_component()`:

1. Find all connected components
2. For any component with > `max_component_size` nodes:
   - Find the minimum score edge in the component
   - Remove all edges with score < (min_score + 0.02)
3. Repeat until all components are within the size limit

The `cosine_delta = 0.02` constant matches the original workflow.

### Step 4 — Label components and export

1. Enumerate connected components (1-indexed)
2. Set `component` attribute on each node and edge
3. Export with `nx.write_graphml(G, output_path, infer_numeric_types=True)`

## Progress Output

Print to stdout during execution:
- Number of edges loaded
- Number of nodes
- Edge count after top-K filtering
- Edge count after each round of component pruning
- Final number of components
- Output file path

## Dependencies

- `pandas` — parquet reading
- `networkx` — graph operations and GraphML export
- `pyarrow` — parquet engine for pandas

## File Location

Single file: `build_network.py` in the project root directory.

## Design Decisions

- **`nx.Graph` instead of `nx.MultiGraph`:** The original workflow uses MultiGraph to support multiple edge types (Cosine, IonIdentity, etc.). We only have one edge type, so a simple Graph suffices and is more memory efficient.
- **Standalone script:** No dependency on the Classical_Networking_Workflow import chain (`ming_fileio_library`, `constants_network`, etc.). The filtering logic is straightforward (~80 lines) and directly ported.
- **Full memory load:** User confirmed ~50 GB memory is available. No chunked processing needed for now.
