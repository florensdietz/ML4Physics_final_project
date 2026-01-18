import os
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def load_train_val_loaders(
    data_dir: str,
    nodes_file: str = "nodes_t.pt",
    vel_file: str = "vel_next.pt",
    edge_index_file: str = "edge_index.pt",
    batch_size: int = 1,
    val_frac: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader]:
    """
    Load training and validation DataLoaders for Graph Network physics learning.

    Parameters
    ----------
    data_dir : str
        Directory containing .pt files

    nodes_file : str
        File containing node states at time t
        Shape: [num_samples, num_nodes, 5]

    vel_file : str
        File containing true velocities at time t+1
        Shape: [num_samples, num_nodes, 2]

    edge_index_file : str
        File containing edge index tensor
        Shape: [2, E]

    batch_size : int
        Number of graphs per batch (usually 1 for variable-size graphs)

    val_frac : float
        Fraction of samples used for validation

    seed : int
        Random seed for reproducibility

    device : str
        'cpu' or 'cuda'

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """

    # Load tensors
    nodes_path = os.path.join(data_dir, nodes_file)
    vel_path = os.path.join(data_dir, vel_file)
    edge_path = os.path.join(data_dir, edge_index_file)

    nodes = torch.load(nodes_path, map_location="cpu").float()
    vel_next = torch.load(vel_path, map_location="cpu").float()
    edge_index = torch.load(edge_path, map_location="cpu").long()

    # Shape checks
    if nodes.ndim != 3:
        raise ValueError(
            f"Expected nodes to have shape [T, N, 5], got {tuple(nodes.shape)}"
        )

    if vel_next.ndim != 3:
        raise ValueError(
            f"Expected vel_next to have shape [T, N, 2], got {tuple(vel_next.shape)}"
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index to have shape [2, E], got {tuple(edge_index.shape)}"
        )

    # Dataset contains only per-sample tensors
    dataset = TensorDataset(nodes, vel_next)

    # Train / validation split
    torch.manual_seed(seed)
    num_samples = len(dataset)
    n_val = int(val_frac * num_samples)
    n_train = num_samples - n_val

    train_data, val_data = random_split(dataset, [n_train, n_val])

    # DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Wrap edge_index so it can be yielded with every batch
    def attach_edge_index(loader):
        for nodes_t, vel_next_true in loader:
            yield (
                nodes_t.squeeze(0).to(device),
                vel_next_true.squeeze(0).to(device),
                edge_index.to(device),
            )

    return attach_edge_index(train_loader), attach_edge_index(val_loader)
