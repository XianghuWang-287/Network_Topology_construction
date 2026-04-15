#!/usr/bin/env python3

import argparse
import os
import sys

import networkx as nx
import pandas as pd


def make_default_output_path(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_network.graphml"


_COLUMN_PATTERNS = {
    "id1": ["spectrum_id_1", "clusterid1", "id1", "node1", "source", "scan1"],
    "id2": ["spectrum_id_2", "clusterid2", "id2", "node2", "target", "scan2"],
    "score": ["cosine_similarity", "cosine", "score", "similarity", "weight", "mzscore", "otherscore"],
}


def _match_column(df_columns, role):
    """Match a dataframe column to a role (id1, id2, score) using common patterns."""
    col_lower_map = {c.lower().strip(): c for c in df_columns}
    for pattern in _COLUMN_PATTERNS[role]:
        if pattern in col_lower_map:
            return col_lower_map[pattern]
    return None


def resolve_columns(df, col_id1=None, col_id2=None, col_score=None):
    """Resolve column names: use explicit args if given, otherwise auto-detect."""
    resolved = {}
    for role, explicit in [("id1", col_id1), ("id2", col_id2), ("score", col_score)]:
        if explicit:
            if explicit not in df.columns:
                raise ValueError(f"Column '{explicit}' not found. Available: {list(df.columns)}")
            resolved[role] = explicit
        else:
            matched = _match_column(df.columns, role)
            if matched is None:
                raise ValueError(
                    f"Cannot auto-detect '{role}' column. "
                    f"Available: {list(df.columns)}. "
                    f"Use --col_{role} to specify explicitly."
                )
            resolved[role] = matched
    return resolved["id1"], resolved["id2"], resolved["score"]


def load_parquet_to_graph(parquet_path, col_id1=None, col_id2=None, col_score=None):
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} edges")
    print(f"  Columns: {list(df.columns)}")

    id1_col, id2_col, score_col = resolve_columns(df, col_id1, col_id2, col_score)
    print(f"  Matched: id1='{id1_col}', id2='{id2_col}', score='{score_col}'")

    G = nx.Graph()

    nodes = set(df[id1_col]).union(set(df[id2_col]))
    G.add_nodes_from(nodes)
    print(f"  {len(nodes)} nodes")

    edges = list(zip(
        df[id1_col].astype(str),
        df[id2_col].astype(str),
        [{"score": float(s)} for s in df[score_col]],
    ))
    G.add_edges_from(edges)
    print(f"  {len(G.edges())} edges in graph")

    return G


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
    parser.add_argument(
        "--col_id1",
        default=None,
        help="Column name for node ID 1 (auto-detected if not specified)",
    )
    parser.add_argument(
        "--col_id2",
        default=None,
        help="Column name for node ID 2 (auto-detected if not specified)",
    )
    parser.add_argument(
        "--col_score",
        default=None,
        help="Column name for edge score (auto-detected if not specified)",
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
    print()

    G = load_parquet_to_graph(args.input_parquet, args.col_id1, args.col_id2, args.col_score)
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


if __name__ == "__main__":
    main()
