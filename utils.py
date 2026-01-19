from typing import Tuple
from torch.utils.data import DataLoader, random_split, Dataset
import os
import torch



class GNPhysicsDataset(Dataset):
    def __init__(
        self,
        nodes: torch.Tensor,       # [T, N, 5]
        vel_next: torch.Tensor,    # [T, N, 2]
        edge_index: torch.Tensor,  # [2, E]
    ):
        assert nodes.shape[0] == vel_next.shape[0]
        self.nodes = nodes
        self.vel_next = vel_next
        self.edge_index = edge_index

    def __len__(self) -> int:
        return self.nodes.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.nodes[idx],       # [N, 5]
            self.vel_next[idx],    # [N, 2]
            self.edge_index,       # [2, E]
        )

def load_train_val_loaders(
    data_dir: str,
    nodes_file: str = "nodes_t.pt",
    vel_file: str = "vel_updated_true.pt",
    edge_index_file: str = "edge_index.pt",
    batch_size: int = 1,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:

    nodes = torch.load(os.path.join(data_dir, nodes_file), weights_only=False).float()
    vel_next = torch.load(os.path.join(data_dir, vel_file), weights_only=False).float()
    edge_index = torch.load(os.path.join(data_dir, edge_index_file), weights_only=False).long()

    # Shape checks
    if nodes.ndim != 3:
        raise ValueError(f"nodes must be [T, N, 5], got {nodes.shape}")
    if vel_next.ndim != 3:
        raise ValueError(f"vel_next must be [T, N, 2], got {vel_next.shape}")
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2, E], got {edge_index.shape}")

    dataset = GNPhysicsDataset(nodes, vel_next, edge_index)

    torch.manual_seed(seed)
    n_total = len(dataset)
    n_val = int(val_frac * n_total)
    n_train = n_total - n_val

    train_data, val_data = random_split(dataset, [n_train, n_val])

    def collate_fn(batch):
        """
        Custom collate function for batching GN datasets.
        Keeps edge_index consistent and avoids extra batch dimensions.
        """
        nodes_batch, vel_batch, _ = zip(*batch)  # ignore batch edge_index
        nodes = nodes_batch[0]   # batch_size=1 -> shape [N, 5]
        vel_next = vel_batch[0]  # shape [N, 2]
        return nodes, vel_next, edge_index

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader
