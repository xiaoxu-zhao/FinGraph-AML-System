import torch
import pytest
from fingraph.core.model import AML_GraphNetwork

def test_model_architecture():
    """
    Test that the model initializes correctly and produces output of the expected shape.
    """
    in_channels = 16
    hidden_channels = 32
    out_channels = 2
    heads = 2
    
    model = AML_GraphNetwork(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads
    )
    
    # Create dummy graph data
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn((num_nodes, in_channels))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn((num_edges, 1)) # 1 feature (Amount)
    
    # Forward pass
    out = model(x, edge_index, edge_attr)
    
    # Assertions
    assert out.shape == (num_nodes, out_channels), f"Expected output shape {(num_nodes, out_channels)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"
