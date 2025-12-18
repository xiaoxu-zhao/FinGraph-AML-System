import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATv2Conv
from typing import Tuple, Optional
from fingraph.utils.logger import get_logger

logger = get_logger(__name__)

class AML_GraphNetwork(torch.nn.Module):
    """
    Graph Neural Network for Anti-Money Laundering detection using GATv2.
    
    Architecture:
        - GATv2Conv Layer 1 (Multi-head attention)
        - Batch Normalization
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
            out_channels (int): Number of output classes.
            heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        logger.info(f"Initializing AML_GraphNetwork with {heads} heads and {hidden_channels} hidden channels.")
        
        self.dropout_p = dropout
        
        # Layer 1: Multi-head attention
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=1)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        
        # Layer 2: Aggregation (heads=1 for final representation before classifier)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1)
        self.bn2 = BatchNorm1d(hidden_channels)
        
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
        """
        try:
            # Layer 1
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
            x = self.bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            # Layer 2
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
            x = self.bn2(x)
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
