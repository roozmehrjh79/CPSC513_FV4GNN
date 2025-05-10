# ----- Imports ----- #
import time
import torch
import numpy as np
from utils import *
from gnn_models import GCN
from memory_profiler import memory_usage
from torch_geometric.datasets import KarateClub
from maraboupy import Marabou


# ----- User-defined parameters ----- #
DEBUG = False
FORCE_USE_CPU = True
NETWORK_NAME = "Equivalent FFNN for 2-layered GCN with ReLU for input graph with stats: |V| = {num_nodes}, |E| = {num_edges}"
PYTORCH_MODEL_NAME = "gcn_karate.pth"
NNET_FILENAME = ""
INF = 1e9


# ----- Export PyTorch model to .nnet format based on input graph ----- #
def gen_properties(mask_idx: int = 0, b: int = 1):
    # Checking if GPU is available
    global FORCE_USE_CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if FORCE_USE_CPU:
        device = torch.device("cpu")

    # Load dataset
    dataset = KarateClub()
    data = dataset[0].to(device)

    # Set subgraph mask from original dataset
    # Feel free to modify based on this image:
    # https://miro.medium.com/v2/resize:fit:720/format:webp/1*gPUxh53IHnS0lnnks8_L3Q.png
    # Note that node labels correspond to the following colors:
    # 0: green, 1: blue, 2: purple, 3: red
    mask_dict = {
        0: [0, 17, 19],
        1: [8, 1, 30, 2],
        2: [4, 5, 6, 10, 16],
        3: [0, 10, 5, 16, 6, 4],
        4: [24, 25, 28, 31, 33, 13, 2],
        5: [0, 1, 2, 3, 7, 9, 11, 12, 13, 17, 19, 21],
        6: [0, 1, 2, 8, 30, 33, 5, 16, 6, 31, 24, 25],
        7: [8, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33],
        8: [i for i in range(34) if i not in [4, 5, 6, 10, 16]],
        9: [i for i in range(34)],
    }

    mask = mask_dict[mask_idx]

    # Prepare for conversion & monitor runtime and memory consumption
    print("--- Exporting graph to .nnet format -----")
    print("Starting graph conversion...")
    time_start = time.time()

    # Construct subgraph
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    node_features = data.x[mask, :]
    edge_index = [[], []]
    edge_attr = []
    for e in range(data.edge_index.shape[1]):
        src, dst = data.edge_index[0][e], data.edge_index[1][e]
        if src in mask and dst in mask:
            new_src, new_dst = mask.index(src), mask.index(dst)
            edge_index[0].append(new_src)
            edge_index[1].append(new_dst)
            edge_attr.append(data.edge_attr[e] if data.edge_attr is not None else 1.0)
    num_edges = len(edge_attr)
    num_nodes = len(mask)
    input_dim = num_nodes * num_features

    ##### TEST #####
    EXPORT_PROP_FILE = f"gcn_karate_V{num_nodes}_E{num_edges}_b{b}.vgnnlib"
    with open(f"./properties/{EXPORT_PROP_FILE}", "w") as f:
        f.write(
            f"; Spec file for KarateClub subgraph |V|={num_nodes}, |E|={num_edges}\n\n"
        )
        f.write("; Graph structure\n")
        f.write(f"(declare-num-nodes {num_nodes})\n")
        f.write(f"(declare-num-edges {num_edges})\n")
        f.write(f"(declare-num-features {num_features})\n")
        f.write(f"(declare-num-classes {num_classes})\n")
        f.write("\n")

        for e in range(num_edges):
            # Edges go from source to destination
            f.write(
                f"(declare-edge X_{edge_index[0][e]} X_{edge_index[1][e]} {edge_attr[e]})\n"
            )
        f.write("\n")

        f.write("; Declaring input and output feature vectors\n")
        for n in range(num_nodes):
            for feat in range(num_features):
                f.write(f"(declare-const X_{n}_{feat} Real)\n")
        f.write("\n")

        for n in range(num_nodes):
            for c in range(num_classes):
                f.write(f"(declare-const Y_{n}_{c} Real)\n")
        f.write("\n")

        f.write("; Input constraints\n")
        for n in range(num_nodes):
            f.write(f"; Node {n}: All features in (0, 1)\n")
            for feat in range(num_features):
                f.write(f"(assert <= X_{n}_{feat} 1)\n")
                f.write(f"(assert >= X_{n}_{feat} 0)\n")
            f.write("\n")

        f.write("; Output constraints: All pre-softmax elements in (-2^b, 2^b-1)\n")
        for n in range(num_nodes):
            for c in range(num_classes):
                f.write(f"(assert <= Y_{n}_{c} {2**b - 1})\n")
                f.write(f"(assert >= Y_{n}_{c} {-(2**b)})\n")


# ----- Main program ----- #
if __name__ == "__main__":
    mask_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b_list = [11, 12, 12, 13, 13, 15, 15, 15, 17, 17]
    for i in range(len(b_list)):
        gen_properties(mask_idx=mask_indexes[i], b=1)
        gen_properties(mask_idx=mask_indexes[i], b=b_list[i])
    # mem_usage = memory_usage(gnn_to_nnet)
    # print('Maximum memory usage (graph to .nnet conversion): %s' % max(mem_usage))
