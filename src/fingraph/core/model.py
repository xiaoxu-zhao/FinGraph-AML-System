import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv
from typing import Tuple, Optional
from fingraph.utils.logger import get_logger

logger = get_logger(__name__)

class AML_GraphNetwork(torch.nn.Module):
    """
    Graph Neural Network for Anti-Money Laundering detection using GATv2.
    
    Architecture:
        - GATv2Conv Layer 1 (Multi-head attention)
        - ReLU + Dropout
        - GATv2Conv Layer 2 (Single-head aggregation)
        - ReLU + Dropout
        - Linear Classification Head
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        heads: int = 4,
        dropout: float = 0.5
    ):
        """
        Initialize the GATv2 model.

        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output classes (e.g., 2 for binary classification).
            heads (int): Number of attention heads for the first layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        logger.info(f"Initializing AML_GraphNetwork with {heads} heads and {hidden_channels} hidden channels.")
        
        self.dropout_p = dropout
        
        # Layer 1: Multi-head attention
        # edge_dim=1 allows us to pass transaction amounts as edge features
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=1)
        
        # Layer 2: Aggregation (heads=1 for final representation before classifier)
        # Input dim is hidden_channels * heads because conv1 concatenates heads
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1)
        
        # Classification Head
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix [num_edges, edge_dim].
            return_embedding (bool): If True, returns (logits, embeddings).

        Returns:
            torch.Tensor: Logits [num_nodes, out_channels] OR (logits, embeddings)
        """
        try:
            # Layer 1
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            # Layer 2
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            
            # Capture embedding before final dropout/classifier
            embedding = x
            
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            # Classifier
            out = self.classifier(x)
            
            if return_embedding:
                return out, embedding
            return out
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise
