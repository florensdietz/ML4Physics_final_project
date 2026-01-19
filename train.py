
import torch
import torch.nn as nn
from typing import Tuple, List
from torch.utils.data import DataLoader
from tqdm import tqdm



def TrainingAlgorithm(model: nn.Module,
                      train_loader: DataLoader,   
                      val_loader: DataLoader,
                      num_epochs: int = 10,
                      learning_rate: float = 1e-3,
                      device: str = "cpu"
                      ) -> Tuple[List[float], List[float]]:

    model = model.to(device)
    model.train()        # training mode

    train_losses = []    # track training loss
    val_losses = []      # track validation loss

    # choosing omptimizer and loss fucntion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)   
    loss_fn = nn.L1Loss()


    for epoch in range(num_epochs):

        total_train_loss = 0.0


        for batch in train_loader:
            nodes_t, vel_next_true, edge_index = batch

            nodes_t = nodes_t.to(device)               # [N, 5]
            vel_next_true = vel_next_true.to(device)   # [N, 2]
            edge_index = edge_index.to(device)         # [2, E]

            optimizer.zero_grad()

            # Forward pass
            updated_nodes = model(nodes_t, edge_index)

            # Extract predicted velocity at t+1
            vel_next_pred = updated_nodes[:, 2:4]

            # Loss only on velocity updates
            loss = loss_fn(vel_next_pred, vel_next_true)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        # Add training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Add validation loss
        avg_val_loss = evaluate(model, val_loader, device=device)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Avg Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
    torch.save(model.state_dict(), "model_weights.pth")

    return train_losses, val_losses
    

def evaluate(model: nn.Module,
             val_loader: DataLoader,
             device: str = "cpu"
             ) -> float:

    model.eval()

    loss_fn = nn.L1Loss()   
    total_val_loss = 0.0

    with torch.no_grad():
        for nodes_t, vel_next_true, edge_index in val_loader:

            nodes_t = nodes_t.to(device).float()
            vel_next_true = vel_next_true.to(device).float()
            edge_index = edge_index.to(device)

            # Forward pass
            updated_nodes = model(nodes_t, edge_index)

            # Extract predicted velocities at t+1
            vel_next_pred = updated_nodes[:, 2:4]

            # Compute loss
            loss = loss_fn(vel_next_pred, vel_next_true)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    return avg_val_loss

