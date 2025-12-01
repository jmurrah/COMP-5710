"""
COMP-5710 Workshop 14: ANGR
Author: Jacob Murrah
Date: 12/5/2025
"""

import angr
import os
from pathlib import Path
from rich.tree import Tree
from rich.console import Console

BINARY_DIR = Path("binaries")

def write_graph_metric_outputs_to_file(file_path, cfg_metrics):
    print("hello")
        

def try1(graph):
    lines = []
    for src, dst in graph.edges:
        # src and dst are CFGNode-like; .addr is the basic block address
        lines.append(f"{hex(src.addr)} -> {hex(dst.addr)}")
    cfg_text = "\n".join(lines)
    print("=== Control Flow Graph (Edge List) ===")
    print(cfg_text)

def try2(cfg):
    print("=== Control Flow Graph (Text Format) ===")
    
    # Sort nodes by address so the output is linear and readable
    # angr CFG nodes are objects, so we sort by their .addr property
    print(dir(cfg.graph.nodes))
    sorted_nodes = sorted(cfg.graph.nodes, key=lambda n: n.addr)

    for node in sorted_nodes:
        # Get the function name if available, otherwise just use the address
        name = node.name if node.name else "unknown_block"
        print(f"[{hex(node.addr)}] <{name}> (Size: {node.size})")

        # Get successors (children) of this node
        successors = list(cfg.graph.successors(node))
        
        if not successors:
            print("    --> (Terminal Block / No known successors)")
        
        for succ in successors:
            # We look up edge data to find out *how* we get there (Jump, Call, Ret)
            edge_data = cfg.graph.get_edge_data(node, succ)
            jump_kind = edge_data.get('jumpkind', 'Ijk_Boring') # 'Ijk_Boring' is a standard jump
            
            print(f"    --> {jump_kind} to {hex(succ.addr)}")
        
        # Add a newline for separation
        print("")


def create_text_formatted_control_flow_graph(graph):
    try1(graph.graph)

    print("GVBFKUWGWDOHDOIWKHDIOWUJDPOWUDHIOUWDHWIODHWOIUDHWOIDUHWOIUDWHODHOWD")

    try2(graph)

def get_crontrol_flow_graph_metrics(project):
    metrics = {}
    graph_cfgfast = project.analyses.CFGFast()
    graph = graph_cfgfast.graph

    metrics["text_cfg"] = create_text_formatted_control_flow_graph(graph_cfgfast)
    metrics["num_nodes"] = len(graph.nodes)
    metrics["num_edges"] = len(graph.edges)

    print(len(project.loader.all_objects))
    memory_ranges = []
    for object in project.loader.all_objects:
        # print(object)
        # print(dir(object))
        
        min_addr = object.min_addr
        max_addr = object.max_addr

        memory_ranges.append({
            "object": object,
            "min_addr": min_addr,
            "max_addr": max_addr,
            "range": (min_addr, max_addr)
        })
    
    metrics["memory_ranges"] = memory_ranges
    metrics["heap_mem_addr"] = project.factory.entry_state().heap.heap_base

    return metrics
    

def main():
    binary_metric_outputs = []
    for binary_path in BINARY_DIR.iterdir():
        angr_proj = angr.Project(str(binary_path), auto_load_libs=False)
        binary_metric_outputs.append(get_crontrol_flow_graph_metrics(angr_proj))
        break

    write_graph_metric_outputs_to_file("text_results.txt", binary_metric_outputs)
    


if __name__ == "__main__":
    main()
