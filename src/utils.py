# ----- Imports ----- #
import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from typing import Tuple, List, Dict


# ----- Utility functions ----- #
def agr_gcn_to_ff(
    num_nodes: int, edge_indexes: Tensor, edge_weights: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Converts the node-wise aggregation of a graph convolution (GCN) layer into
    its equivalent feed-forward (FF) layer form. The attributes of the input
    graph are as follows:
    - `V`: Number of nodes in the graph.
    - `E`: Number of edges in the graph.

    This function was written based on the formula available at:
    https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.nn.conv.GCNConv.html

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the input graph (a.k.a. `V`).
    edge_indexes : Tensor
        Edge index matrix of the input graph in Coordinate (COO) format. Must be
        of shape `(2, E)` and contain values in range `[0, V-1]`.
    edge_weights : Tensor
        Edge weights of the input graph. Must be of shape `(E)`.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing the FF-equivalent weights and biases.
    """
    # Checking if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Initialization
    V = num_nodes
    E = len(edge_weights.shape)

    # Initialize FF-equivalent weights & biases
    weights_ff = torch.eye(V).to(device)
    biases_ff = torch.zeros(V).to(device)

    # Get weighted degree of each node in the graph
    # Also partially calculate the new weights
    degrees = torch.ones(V).to(device)
    for e in range(E):  # traverse through all edges
        src, dst = int(edge_indexes[0][e]), int(edge_indexes[1][e])
        if src != dst:
            weights_ff[dst][src] += edge_weights[e]
            degrees[dst] += edge_weights[e]
        else:
            weights_ff[dst][src] *= edge_weights[e]

    # Complete the calculations for the new weight matrix
    for e in range(E):  # traverse through all edges
        src, dst = int(edge_indexes[0][e]), int(edge_indexes[1][e])
        weights_ff[src][dst] /= torch.sqrt(degrees[src] * degrees[dst])

    return weights_ff, biases_ff


def gcn_to_ff(
    num_nodes: int,
    edge_indexes: Tensor,
    edge_weights: Tensor,
    weights_gcn: Tensor,
    biases_gcn: Tensor,
    w_agr2ff: Tensor | None = None,
    b_agr2ff: Tensor | None = None,
) -> Tuple[Tensor, Tensor]:
    """
    Generates the FF-equivalent network of a GCN layer. The attributes of the
    input graph are as follows:
    - `V`: Number of nodes in the graph.
    - `E`: Number of edges in the graph.
    - `F_N`: Number of input node features.
    - `F_H`: Number of output node features (`H` stands for hidden).

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the input graph (a.k.a. `V`).
    edge_indexes : Tensor
        Edge index matrix of the input graph in Coordinate (COO) format. Must be
        of shape `(2, E)` and contain values in range `[0, V-1]`.
    edge_weights : Tensor
        Edge weights of the input graph. Must be of shape `(E)`.
    weights_gcn : Tensor
        Weight matrix of the GCN layer. Must be of shape `(F_H, F_N)`.
    biases_gcn : Tensor
        Bias vector of the GCN layer. Must be of shape `(F_H)`.
    w_agr2ff : Tensor | None, optional
        FF-equivalent weight matrix of the GCN aggregation layer. Must be of
        shape `(V, V)`. Defaults to `None`.
    b_agr2ff : Tensor | None, optional
        FF-equivalent bias vector of the GCN aggregation layer. Must be of shape
        `(V)`. Defaults to `None`.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing the FF-equivalent weights and biases.
    """
    # Checking if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Initialization
    V = num_nodes
    F_H, F_N = weights_gcn.shape
    input_dim = V * F_N
    output_dim = V * F_H

    # Get the FF-equivalent node-wise aggregation weights & biases
    if w_agr2ff is None or b_agr2ff is None:
        w_agr2ff, b_agr2ff = agr_gcn_to_ff(V, edge_indexes, edge_weights)

    # Initialize FF-equivalent new weights & biases
    weights_agr2ff = torch.zeros((V, F_N, V, F_N)).to(device)
    biases_agr2ff = torch.zeros((V, F_N)).to(device)
    weights_gcn2ff = torch.empty((V, F_H, V, F_N)).to(device)
    biases_gcn2ff = torch.empty((V, F_H)).to(device)

    # Fill in the weights & biases
    for dst in range(V):
        for src in range(V):
            for f in range(F_N):
                weights_agr2ff[dst, f, src, f] = w_agr2ff[dst][src]
            weights_gcn2ff[dst, :, src, :] = weights_gcn
        biases_agr2ff[dst, :] = b_agr2ff[dst]
        biases_gcn2ff[dst, :] = biases_gcn

    # Break down the dimensions of the new weights & biases
    weights_agr2ff = torch.reshape(weights_agr2ff, (input_dim, input_dim))
    biases_agr2ff = torch.reshape(biases_agr2ff, (input_dim,))
    weights_gcn2ff = torch.reshape(weights_gcn2ff, (output_dim, input_dim))
    biases_gcn2ff = torch.reshape(biases_gcn2ff, (output_dim,))

    # Calculate the final combined weights & biases
    weights_ff = torch.matmul(weights_gcn2ff, weights_agr2ff)
    biases_ff = torch.matmul(weights_gcn2ff, biases_agr2ff) + biases_gcn2ff

    return weights_ff, biases_ff


def export_to_nnet(
    input_stats: Dict[str, Tensor],
    output_stats: Dict[str, float],
    weights: List[Tensor],
    biases: List[Tensor],
    filename: str,
    network_name: str | None = None,
):
    """
    Generates a `.nnet` file suitable for Reluplex from a set of input/output
    stats, weights, and biases, belonging to a FF-NN.

    Parameters
    ----------
    input_stats : Dict[str, Tensor]
        A dictionary containing statistics of the inputs to the NN. Must have
        the following keys, with each value being a tensor of shape
        `(INPUT_DIM)`:
        - `"min"`: Holds the minimum possible value for each input.
        - `"max"`: Holds the maximum possible value for each input.
        - `"mean"`: Holds the mean value for each input.
        - `"range"`: Holds the range value for each input.

    output_stats : Dict[str, float]
        A dictionary containing statistics of the outputs of the NN. Must have
        the following keys, with each value being a float:
        - `"mean"`: Holds the mean value for all outputs.
        - `"range"`: Holds the range value for all outputs.

    weights : List[Tensor]
        List of weight matrices.
    biases : List[Tensor]
        List of bias vectors. Must have the same length as `weights`.
    filename : str
        Output file name.
    network_name : str | None, optional
        Name of the neural network. Defaults to `None`.
    """
    # Checking for size mismatch
    assert len(weights) == len(
        biases
    ), "Number of weight matrices and bias vectors must be the same."

    # Open the .nnet file and start writing to it
    with open(filename, "w") as f:
        # Header
        if network_name is None:
            f.write("// Neural network file format\n")
        else:
            f.write(f"// {network_name}\n")

        # Initializations
        num_layers = len(weights)
        num_inputs = weights[0].shape[1]
        num_outputs = weights[-1].shape[0]
        max_layer_size = max(num_inputs, max(len(b) for b in biases))

        # Write network architecture
        f.write(f"{num_layers},{num_inputs},{num_outputs},{max_layer_size},\n")
        f.write(f"{num_inputs}," + ",".join(str(len(b)) for b in biases) + ",\n")
        f.write("0,\n")  # unused flag

        # Write input & output statistics
        f.write(
            ",".join(f"{input_stats['min'][i]:.5e}" for i in range(num_inputs)) + ",\n"
        )
        f.write(
            ",".join(f"{input_stats['max'][i]:.5e}" for i in range(num_inputs)) + ",\n"
        )
        f.write(
            ",".join(f"{input_stats['mean'][i]:.5e}" for i in range(num_inputs))
            + f",{output_stats['mean']},\n"
        )
        f.write(
            ",".join(f"{input_stats['range'][i]:.5e}" for i in range(num_inputs))
            + f",{output_stats['range']},\n"
        )

        # Write weights and biases for each layer
        for l in range(num_layers):
            for r in range(weights[l].shape[0]):
                f.write(
                    ",".join(
                        f"{weights[l][r, c]:.5e}" for c in range(weights[l].shape[1])
                    )
                    + ",\n"
                )
            for i in range(len(biases[l])):
                f.write(f"{biases[l][i]:.5e},\n")


def gnn_to_mlp(model_sdict_path: str, specs_file_path: str, export_dir: str):
    """
    Converts a GNN alongside a set of input/output specs to an equivalent MLP.
    The converted network is in ONNX format, with its associated specs in VNNLIB format.

    Parameters
    ----------
    model_sdict_path : str
        Path to state dictionary of the pre-trained GNN.
    specs_file_path: str
        Path to input graph specs file. Must be a plain text file with a
        specific format described later on.
    export_dir : str
        The directory in which the exported files would be saved to.
    """
    # TODO: Load the specs file and configure input graph the same way done in
    # the `verify.py` file.

    # TODO: Load the saved model's .pt file and iterate through all of its
    # layers, converting each GCN layer and ultimately building an MLP network
    # and loading the values into it.

    # TODO: Export the finalized MLP to an ONNX file, and also export
    # equivalent specs to a VNNLIB file.
    
    # Initialize variables
    num_nodes = 0
    num_edges = 0
    num_features = 0
    num_classes = 0
    edge_index = [[], []]
    edge_attr = []
    input_bounds: Dict[int, Dict[str, float]] = {}
    output_bounds: Dict[int, Dict[str, float]] = {}
    
    # Load and parse the specs file
    # TODO: Needs completion
    with open(specs_file_path, "r") as f:
        for line in f:
            if line.strip().startswith("("):
                arg = line.strip()[1:-2].split(" ")
                match arg[0]:  # arg[0] is always the command
                    case "declare-num-nodes":
                        num_nodes = int(arg[1])
                    case "declare-num-edges":
                        num_edges = int(arg[1])
                    case "declare-num-features":
                        num_features = int(arg[1])
                    case "declare-num-classes":
                        num_classes = int(arg[1])
                    case "declare-edge":
                        src = int(arg[1].split('_')[-1])
                        dest = int(arg[2].split('_')[-1])
                        weight = float(arg[3])
                        edge_index[0].append(src)
                        edge_index[1].append(dest)
                        edge_attr.append(weight)
                    case "assert":
                        operator = arg[0]
                        value = float(arg[3])
                        target = arg[2].split("_")
                        node = int(target[1])
                        type = "input" if target[0] == "X" else "output"
                        if type == "input":
                            node = int(target[1])
                            feature = int(target[2])
                            var_idx = node * num_features + feature
                            # TODO: Completion needed
                        else:
                            cls_label = int(target[2])
                            # TODO: Completion needed
    pass
