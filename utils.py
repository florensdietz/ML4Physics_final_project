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



import torch
import torch.nn as nn
from typing import Dict

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: str = "cpu",
    rollout_steps: int = 0,
    dt: float = 1.0,
    update_positions_in_rollout: bool = True,
    strict_rollout_length: bool = True,   # if True: raise if dataset too short for requested rollout
) -> Dict[str, float]:
    """
    Evaluates a GN model.

    Returns:
      - one_step_l1: mean L1 on v_{t+1} for the samples in loader
      - rollout_l1:  mean L1 over a K-step autoregressive rollout (if rollout_steps>0)

    Assumptions:
      - loader yields (nodes_t: [N,5], vel_next_true: [N,2], edge_index: [2,E])
      - node format: (x,y,vx,vy,m) in nodes[:,0:5]
      - model(nodes, edge_index) returns updated_nodes where updated_nodes[:,2:4] is v_{t+1} prediction
    """
    model = model.to(device)
    model.eval()

    l1 = nn.L1Loss(reduction="mean")

    total_one_step = 0.0
    n_batches = 0

    total_rollout = 0.0
    n_rollout_batches = 0

    def step_state(nodes: torch.Tensor, vel_next_pred: torch.Tensor) -> torch.Tensor:
        """
        Build next nodes state from current nodes and predicted next velocity.
        Keeps mass unchanged.
        Optionally updates positions using Euler: x_{t+1} = x_t + v_{t+1}*dt.
        """
        new_nodes = nodes.clone()
        new_nodes[:, 2:4] = vel_next_pred
        if update_positions_in_rollout:
            new_nodes[:, 0:2] = nodes[:, 0:2] + vel_next_pred * dt
        return new_nodes

    # ---- ONE-STEP EVAL ----
    for nodes_t, vel_next_true, edge_index in loader:
        nodes_t = nodes_t.to(device).float()
        vel_next_true = vel_next_true.to(device).float()
        edge_index = edge_index.to(device)

        updated_nodes = model(nodes_t, edge_index)
        vel_next_pred = updated_nodes[:, 2:4]

        total_one_step += l1(vel_next_pred, vel_next_true).item()
        n_batches += 1

    results = {"one_step_l1": total_one_step / max(1, n_batches)}

    # ---- ROLLOUT EVAL (K-step) ----
    if rollout_steps and rollout_steps > 0:
        ds = loader.dataset
        indices = None

        # random_split returns a Subset with .dataset and .indices
        if hasattr(ds, "indices") and hasattr(ds, "dataset"):
            indices = ds.indices
            base_ds = ds.dataset
        else:
            base_ds = ds

        # helper to map "subset index" -> "base dataset item"
        if indices is None:
            T = len(base_ds)
            def get_item(i: int):
                return base_ds[i]
        else:
            T = len(indices)
            def get_item(j: int):
                return base_ds[indices[j]]

        # If dataset is too short for requested rollout, either raise or clamp
        if T < rollout_steps:
            if strict_rollout_length:
                raise ValueError(f"rollout_steps={rollout_steps} > available sequence length T={T}")
            rollout_steps_eff = T
        else:
            rollout_steps_eff = rollout_steps

        # IMPORTANT FIX: +1 so that if T == rollout_steps, start=0 is included
        valid_starts = range(0, T - rollout_steps_eff + 1)

        for start in valid_starts:
            # initial state at time "start"
            nodes_0, _, edge_index0 = get_item(start)
            nodes = nodes_0.to(device).float()
            edge_index0 = edge_index0.to(device)

            rollout_loss = 0.0
            for k in range(rollout_steps_eff):
                # Dataset item i contains nodes_i and vel_{i+1}
                _, vel_true, _ = get_item(start + k)
                vel_true = vel_true.to(device).float()

                updated_nodes = model(nodes, edge_index0)
                vel_pred = updated_nodes[:, 2:4]

                rollout_loss += l1(vel_pred, vel_true).item()

                # advance state autoregressively
                nodes = step_state(nodes, vel_pred)

            total_rollout += rollout_loss / rollout_steps_eff
            n_rollout_batches += 1

        # If no valid starts exist (can happen if T==0), report NaN instead of misleading 0
        if n_rollout_batches == 0:
            results["rollout_l1"] = float("nan")
        else:
            results["rollout_l1"] = total_rollout / n_rollout_batches

    return results
