import torch
import torch.nn as nn
import os


class GraphNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        num_hidden = 128       # number of hidden layers
        node_dim = 5           # number of attributes per node
        msg_dim = 2            # message dimension equals spatial dimesion
        
        # Define MLP representing the message function phi_e between two nodes 
        self.MLP_message_function = nn.Sequential(
            nn.Linear(2 * node_dim, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, msg_dim)
        )
        
        # Define MLP representing the node update function phi_v 
        self.MLP_node_update_function = nn.Sequential(
            nn.Linear(node_dim + msg_dim, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, msg_dim)  
        )

    
    def pooling_function(self,
                         messages: torch.Tensor,
                         receiver_nodes: torch.Tensor,
                         num_nodes: int
                         ) -> torch.Tensor:
        """Pooling function rho_{e->v}: Adds all messages sent to node i. 
        In other terms: all forces that act on body i are added."""

        pooled = torch.zeros(num_nodes, messages.size(1), device=messages.device)
        pooled.index_add_(0, receiver_nodes, messages)

        return pooled
    

    def update_nodes(self,
                     nodes: torch.Tensor,
                     delta_v: torch.Tensor
                     ) -> torch.Tensor:
        """Update the velocities by using Euler integration. 
        The updated velocities define the next timestep."""

        new_nodes = nodes.clone()
        new_nodes[:, 2:4] = nodes[:, 2:4] + delta_v 

        return new_nodes


    def forward(self, 
                nodes: torch.Tensor,       # N x (2D+1) (N = number of bodies, D = dimension, use convention (x,y,v_x,v_y,m) for D=2)
                edge_index: torch.Tensor   # 2 x E (Indices of senders (row 1) and receivers (row 2), E = total number of edges = N(N-1))
                ) -> torch.Tensor:
        
        senders, receivers = edge_index    # extract sender and receiver indices from edge_index
        
        # 1. Prepare inputs for phi_e
        sender_nodes = nodes[senders]
        receiver_nodes = nodes[receivers]   
        edge_inputs = torch.cat([sender_nodes, receiver_nodes], dim=-1)
        
        # 2. Compute messages for every node pair 
        messages = self.MLP_message_function(edge_inputs)   # E x msg_dim
        
        # 3. Aggregate messages (Pooling)
        pooled_messages = self.pooling_function(messages, receivers, nodes.size(0)) # N x msg_dim
        
        # 4. Prepare inputs for phi_v
        node_update_input = torch.cat([nodes, pooled_messages], dim=-1)

        # 5. Compute Node Update
        delta_v = self.MLP_node_update_function(node_update_input) # N x msg_dim
   
        # 6. Euler Integration to update velocities (note for training: updated_nodes also contains positions and masses)
        updated_nodes = self.update_nodes(nodes, delta_v)  # N x (2D+1)
        
        return updated_nodes