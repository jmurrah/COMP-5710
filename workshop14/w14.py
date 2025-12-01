"""
COMP-5710 Workshop 14: ANGR Binary Analysis
Author: Jacob Murrah
Date: 12/5/2025
"""

import angr
import os
import io
import sys
import logging
from pathlib import Path
from rich.tree import Tree
from rich.console import Console

logging.getLogger("angr").setLevel(logging.CRITICAL)
sys.setrecursionlimit(10000)

BINARY_DIR = Path("binaries")
RESULTS_DIR = Path("results")


def write_cfg_metrics_binary_result(cfg_metrics):
    filename = RESULTS_DIR / f"{cfg_metrics['binary_name']}_results.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Results for Binary File: {cfg_metrics['binary_name']}\n")
        f.write("=" * 60 + "\n")

        f.write(f"Number of Nodes: {cfg_metrics['num_nodes']}\n")
        f.write(f"Number of Edges: {cfg_metrics['num_edges']}\n")
        f.write(f"Heap Base Memory Address: {hex(cfg_metrics['heap_mem_addr'])}\n\n")

        f.write("--- Memory Ranges ---\n")
        for mem in cfg_metrics["memory_ranges"]:
            obj_name = os.path.basename(mem["object"].binary)
            f.write(
                f"{obj_name:<15} : {hex(mem['min_addr'])} - {hex(mem['max_addr'])}\n"
            )

        f.write("\n--- Control Flow Graph Visualization ---\n")
        f.write(cfg_metrics["text_cfg"])
        f.write("\n\n")

    print(f"   - Number of Nodes: {cfg_metrics['num_nodes']}")
    print(f"   - Number of Edges: {cfg_metrics['num_edges']}")
    print(f"   - Heap Base Memory Address: {hex(cfg_metrics['heap_mem_addr'])}")
    print(f"   - Memory Ranges:")
    for mem in cfg_metrics["memory_ranges"]:
        obj_name = os.path.basename(mem["object"].binary)
        print(
            f"       {obj_name:<15} : {hex(mem['min_addr'])} - {hex(mem['max_addr'])}"
        )
    print(f"   - Saved result to {filename}")


def build_rich_tree(cfg, node, tree, global_visited):
    successors = sorted(list(cfg.graph.successors(node)), key=lambda n: n.addr)
    for succ in successors:
        edge_data = cfg.graph.get_edge_data(node, succ)
        jump_kind = edge_data.get("jumpkind", "Boring").replace("Ijk_", "")

        func_name = succ.name if succ.name else "block"
        label = f"{hex(succ.addr)} <{func_name}> (via {jump_kind})"

        if succ in global_visited:
            tree.add(f"{label} -> (ref {hex(succ.addr)})")
        else:
            global_visited.add(succ)
            branch = tree.add(label)
            build_rich_tree(cfg, succ, branch, global_visited)


def create_cfg_visualization(cfg, project):
    entry_node = cfg.model.get_any_node(project.entry)
    root = Tree(f"{hex(entry_node.addr)} <{entry_node.name}>")

    build_rich_tree(cfg, entry_node, root, global_visited={entry_node})

    console = Console(file=io.StringIO(), width=150, no_color=True, highlight=True)
    console.print(root)

    return console.file.getvalue()


def get_control_flow_graph_metrics(project, binary_path):
    metrics = {}
    memory_ranges = []

    cfg = project.analyses.CFGFast(normalize=True)
    for obj in project.loader.all_objects:
        memory_ranges.append(
            {"object": obj, "min_addr": obj.min_addr, "max_addr": obj.max_addr}
        )

    metrics["binary_name"] = binary_path.name
    metrics["num_nodes"] = len(cfg.graph.nodes)
    metrics["num_edges"] = len(cfg.graph.edges)
    metrics["text_cfg"] = create_cfg_visualization(cfg, project)
    metrics["memory_ranges"] = memory_ranges
    metrics["heap_mem_addr"] = project.factory.entry_state().heap.heap_base

    return metrics


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("-" * 60)

    for binary_path in BINARY_DIR.iterdir():
        print(f"Processing Binary File: {binary_path.name}")
        angr_proj = angr.Project(str(binary_path), auto_load_libs=False)
        cfg_metrics = get_control_flow_graph_metrics(angr_proj, binary_path)
        write_cfg_metrics_binary_result(cfg_metrics)
        print("-" * 60)

    print(f"\nResults are saved to: {RESULTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
