# Network Topology Construction

Build a molecular network topology from pairwise spectrum similarity scores. Reads a Parquet file of score pairs, applies GNPS-style network filtering, and exports a GraphML file that can be loaded in Cytoscape or any graph analysis tool.

## Requirements

- Python 3.8+
- pandas
- networkx
- pyarrow

```bash
pip install pandas networkx pyarrow
```

## Quick Start

```bash
python build_network.py data/edges.parquet
```

This reads `data/edges.parquet` and writes `data/edges_network.graphml`.

## Usage

```
python build_network.py <input_parquet> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input_parquet` | *(required)* | Path to input Parquet file |
| `-o`, `--output` | `<input_dir>/<input_name>_network.graphml` | Output GraphML file path |
| `--top_k` | `10` | Max edges per node in mutual top-K filtering |
| `--max_component_size` | `100` | Max nodes per connected component |
| `--col_id1` | *(auto-detect)* | Column name for node ID 1 |
| `--col_id2` | *(auto-detect)* | Column name for node ID 2 |
| `--col_score` | *(auto-detect)* | Column name for edge score |

### Examples

```bash
# Default parameters (top_k=10, max_component_size=100)
python build_network.py data/edges.parquet

# Custom output path and parameters
python build_network.py data/edges.parquet -o output/network.graphml --top_k 15 --max_component_size 200

# Specify column names explicitly
python build_network.py data/my_scores.parquet --col_id1 "SpecID_A" --col_id2 "SpecID_B" --col_score "my_score"
```

## Input Format

The input Parquet file must contain three columns representing pairwise scores between nodes. Column names are auto-detected (case-insensitive) from common patterns:

| Role | Auto-detected names |
|------|-------------------|
| Node ID 1 | `spectrum_id_1`, `clusterid1`, `id1`, `node1`, `source`, `scan1` |
| Node ID 2 | `spectrum_id_2`, `clusterid2`, `id2`, `node2`, `target`, `scan2` |
| Score | `cosine_similarity`, `cosine`, `score`, `similarity`, `weight`, `mzscore`, `otherscore` |

If your column names don't match any of these patterns, use `--col_id1`, `--col_id2`, and `--col_score` to specify them explicitly.

## Output Format

The output is a [GraphML](http://graphml.graphdrawing.org/) file containing the filtered network:

- **Node attributes:** `component` (integer index of the connected component the node belongs to)
- **Edge attributes:** `score` (the pairwise similarity score), `component` (connected component index)

## Algorithm

The filtering algorithm is based on the [GNPS Classical Molecular Networking](https://gnps.ucsd.edu/) workflow:

1. **Load** all pairwise scores into a graph
2. **Top-K mutual filtering** -- For each node, keep only its top K highest-scoring edges. An edge is retained only if it appears in the top K for *both* of its endpoints (mutual filtering).
3. **Component size filtering** -- If any connected component exceeds `max_component_size`, iteratively remove its lowest-scoring edges (within a delta of 0.02) until all components are within the size limit.
4. **Label and export** -- Assign a component index to each node and edge, remove isolated nodes, and write the result as GraphML.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
