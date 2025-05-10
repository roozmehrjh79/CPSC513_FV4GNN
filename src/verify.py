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
def gnn_to_nnet():
    # Checking if GPU is available
    global FORCE_USE_CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    if FORCE_USE_CPU:
        device = torch.device('cpu')
        
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
        9: [i for i in range(34)]
    }
    mask = mask_dict[0]

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
    
    # Fill in the parameters using V & E
    NETWORK_NAME = f"Equivalent FFNN for 2-layered GCN with ReLU for input graph with stats: |V| = {num_nodes}, |E| = {num_edges}"
    NNET_FILENAME = f"gcn_karate_V{num_nodes}_E{num_edges}.nnet"
    
    # Convert to tensor
    edge_index = torch.tensor(edge_index, device=device)
    edge_attr = torch.tensor(edge_attr, device=device)
    
    # Get FF-equivalent weights & biases for GCN aggregation
    w_agr2ff, b_agr2ff = agr_gcn_to_ff(
        num_nodes=num_nodes,
        edge_indexes=edge_index,
        edge_weights=edge_attr
    )
    
    # Load weights & biases from trained model
    model = GCN(
        in_channels=num_features,
        hidden_channels=8,
        out_channels=num_classes,
    )
    model.eval()
    global PYTORCH_MODEL_NAME
    model.load_state_dict(torch.load(f"./models/torch/{PYTORCH_MODEL_NAME}", weights_only=True))
    w_gcn1, b_gcn1 = model.state_dict()['conv1.lin.weight'], model.state_dict()['conv1.bias']
    w_gcn2, b_gcn2 = model.state_dict()['conv2.lin.weight'], model.state_dict()['conv2.bias']
    
    # Get FF-equivalent weights & biases for the entire GCN network
    w_ff1, b_ff1 = gcn_to_ff(
        num_nodes=num_nodes,
        edge_indexes=edge_index,
        edge_weights=edge_attr,
        weights_gcn=w_gcn1,
        biases_gcn=b_gcn1,
        w_agr2ff=w_agr2ff,
        b_agr2ff=b_agr2ff
    )
    w_ff2, b_ff2 = gcn_to_ff(
        num_nodes=num_nodes,
        edge_indexes=edge_index,
        edge_weights=edge_attr,
        weights_gcn=w_gcn2,
        biases_gcn=b_gcn2,
        w_agr2ff=w_agr2ff,
        b_agr2ff=b_agr2ff
    )
    
    # Print spacetime statistics
    print("Graph conversion is done!")
    print(f"-> Elapsed time: {time.time() - time_start}")
    
    # Define input & output statistics (for normalization purposes)
    input_stats = {
        'min': torch.zeros(input_dim, device=device),
        'max': torch.ones(input_dim, device=device),
        'mean': 0.5 * torch.ones(input_dim, device=device),
        'range': torch.ones(input_dim, device=device),
    }
    output_stats = {
        'mean': 1.5,
        'range': 3
    }
    
    # Export to .nnet format
    export_to_nnet(
        input_stats=input_stats,
        output_stats=output_stats,
        weights=[w_ff1, w_ff2],
        biases=[b_ff1, b_ff2],
        filename=f"./models/nnet/{NNET_FILENAME}",
        network_name=NETWORK_NAME
    )
    
# ----- Verify using Marabou ----- #
def verify_nnet():
    # Setup
    global NNET_FILENAME
    global INF
    options = Marabou.createOptions(verbosity = 0)
    nnet_file_path = f"./models/nnet/{NNET_FILENAME}"
    network = Marabou.read_nnet(nnet_file_path)

    # Get input & output dimensions
    input_size = np.shape(network.inputVars[0])[1]
    output_size = np.shape(network.outputVars[0])[1]

    # Set constraints
    upper_bounds = [-0.25, -0.5, -1.] + [-2**i for i in range(1, 13)]
    lower_bounds = [0.25, 0.5, 1.] + [2**i-1 for i in range(1, 13)]

    # Verify for each pair of upper & lower bounds over all outputs
    print("--- Now running Marabou -----")
    with open(f"./results/{NNET_FILENAME}.txt", 'w') as f:
        f.write(">>> Verifying network using Marabou <<<\n")
        f.write(f"File path: ./models/nnet/{NNET_FILENAME}\n")
        f.write(f"Input size: {input_size} | Output size: {output_size}\n")
        f.write("Note: For any query, UNSAT = True and SAT = False.\n\n")
        for b in range(len(upper_bounds)):
            # Initialization
            elapsed_time = 0.
            exit_code = 'unsat'
            f.write(f"Solving query: Check if all outputs fall within range ({upper_bounds[b]}, {lower_bounds[b]})\n")
            f.write("----------")
            
            # Check for upper & lower bounds separately
            for mode in ['lower', 'upper']:
                # Check for each variable separately
                for v in range(output_size):
                    # Set lower / upper bound & solve
                    if mode == 'lower':
                        network.setLowerBound(network.outputVars[0][0][v], lower_bounds[b])
                        f.write(f"\nSolving sub-query: Check if output {v} falls within range (-inf, {lower_bounds[b]})\n")
                        exit_code, vals, stats = network.solve(options=options)
                        network.setLowerBound(network.outputVars[0][0][v], -INF)  # reset
                    else:
                        network.setUpperBound(network.outputVars[0][0][v], upper_bounds[b])
                        f.write(f"\nSolving sub-query: Check if output {v} falls within range ({upper_bounds[b]}, inf)\n")
                        exit_code, vals, stats = network.solve(options=options)
                        network.setUpperBound(network.outputVars[0][0][v], INF)  # reset
                        
                    elapsed_time += stats.getTotalTimeInMicro()
                    if exit_code != 'unsat':
                        break
                
                if exit_code != 'unsat':
                    break
            
            # Write out statistics
            f.write("\n##########\n")
            f.write(f"Query: Check if outputs fall within range ({upper_bounds[b]}, {lower_bounds[b]})\n")
            f.write(f"-> Total execution time: {elapsed_time}ms.\n")
            if exit_code == 'unsat':
                f.write("-> Result: True (UNSAT); all outputs fall within specified ranges.\n")
            elif exit_code == 'sat':
                f.write("-> Result: False (SAT); some outputs fall out of specified ranges.\n")
            else:
                f.write("-> Result: Error or timeout!\n")
            f.write("##########\n\n")
    
    print("----- Verification done -----")
    print(f"Result file generated at: ./results/{NNET_FILENAME}.txt")


# ----- Main program ----- #
if __name__ == "__main__":
    mem_usage = memory_usage(gnn_to_nnet)
    print('Maximum memory usage (graph to .nnet conversion): %s' % max(mem_usage))
    
    #NNET_FILENAME = "gcn_karate_V34_E156.nnet"  # manual override
    #NNET_FILENAME = "gcn_karate_VT_ET.nnet"  # manual override
    #mem_usage = memory_usage(verify_nnet)
    #print('Maximum memory usage (verification with Marabou): %s' % max(mem_usage))