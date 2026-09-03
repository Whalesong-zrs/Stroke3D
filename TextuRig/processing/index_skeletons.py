#!/usr/bin/env python3
"""Convert named skeletons to stable joint indices and a rooted tree."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def split_named_edge(content: str, names: set[str]) -> tuple[str, str] | None:
    for parent in sorted(names, key=len, reverse=True):
        prefix = parent + " "
        if content.startswith(prefix) and content[len(prefix) :] in names:
            return parent, content[len(prefix) :]
    return None


def convert_one(task: tuple[Path, Path]) -> str:
    input_path, output_path = task
    lines = input_path.read_text(encoding="utf-8").splitlines()
    joints: dict[str, tuple[int, list[float]]] = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] == "joints":
            name = " ".join(parts[1:-3])
            if name not in joints:
                joints[name] = (len(joints), [float(value) for value in parts[-3:]])
    if not joints:
        raise ValueError(f"no joints in {input_path}")

    names = set(joints)
    edges: list[tuple[str, str]] = []
    root: str | None = None
    for line in lines:
        line = line.strip()
        if line.startswith("hier "):
            edge = split_named_edge(line[5:], names)
            if edge:
                edges.append(edge)
        elif line.startswith("root "):
            candidate = line[5:]
            if candidate in names:
                root = candidate
    if root is None and edges:
        root = edges[0][0]
    if root is None:
        raise ValueError(f"no valid root or hierarchy in {input_path}")

    graph = nx.Graph()
    graph.add_nodes_from(names)
    graph.add_edges_from(edges)
    component = graph.subgraph(nx.node_connected_component(graph, root))
    tree = nx.minimum_spanning_tree(component)
    queue = deque([root])
    visited = {root}
    directed: list[tuple[str, str]] = []
    while queue:
        parent = queue.popleft()
        for child in tree.neighbors(parent):
            if child not in visited:
                visited.add(child)
                queue.append(child)
                directed.append((parent, child))

    output: list[str] = []
    for _name, (index, xyz) in sorted(joints.items(), key=lambda item: item[1][0]):
        output.append(f"joints joint{index} {' '.join(map(str, xyz))}")
    output.append(f"root joint{joints[root][0]}")
    for parent, child in directed:
        output.append(f"hier joint{joints[parent][0]} joint{joints[child][0]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return input_path.name


def main() -> None:
    args = parse_args()
    tasks = [(path, args.output_dir / path.name) for path in sorted(args.input_dir.glob("*.txt"))]
    if not tasks:
        raise FileNotFoundError(f"no .txt skeletons in {args.input_dir}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(convert_one, tasks), total=len(tasks), desc="skeletons"))


if __name__ == "__main__":
    main()

