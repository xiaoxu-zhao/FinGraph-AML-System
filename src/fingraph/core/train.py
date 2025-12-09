import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from fingraph.core.model import AML_GraphNetwork
from fingraph.data.data_engine import TransactionLoader
from fingraph.data.download import download_data, FILE_NAME_TRANS, DATA_DIR
from fingraph.utils.logger import get_logger

logger = get_logger(__name__)

def train():
    # 1. Ensure Data Exists
    download_data()
    csv_path = DATA_DIR / FILE_NAME_TRANS
    
    # 2. Load Data
    loader = TransactionLoader(str(csv_path))
    data = loader.export_graph_data()
    
    # 3. Split Data (Transductive Split)
    # Create masks for train/val/test
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * 0.7)
    val_size = int(num_nodes * 0.15)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # 4. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Switch to Binary Classification (out_channels=1) for BCEWithLogitsLoss
    model = AML_GraphNetwork(
        in_channels=16, 
        hidden_channels=32,
        out_channels=1,
        heads=4
    ).to(device)
    
    # Moderate learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # 5. Training Loop
    logger.info("Starting training...")
    model.train()
    
    # Keep epochs at 10
    num_epochs = 10
    
    # Handle class imbalance
    # Calculate weight for positive class
    num_neg = (data.y == 0).sum()
    num_pos = (data.y == 1).sum()
    # Use moderate fixed weight (5.0) to prevent oscillation
    pos_weight = torch.tensor([5.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    logger.info(f"Class imbalance: {num_neg} negatives, {num_pos} positives. Using BCE pos_weight={pos_weight.item():.2f}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # BCE expects float targets of shape [N, 1]
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float().unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        # Log every epoch
        # Validation
        model.eval()
        with torch.no_grad():
            # Sigmoid for probability
            probs = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr))
            pred = (probs > 0.5).long().squeeze()
            
            val_mask = data.val_mask
            val_acc = (pred[val_mask] == data.y[val_mask]).sum() / val_mask.sum()
            
            # Also check recall on validation set to see if we are catching any positives
            val_y = data.y[val_mask].cpu().numpy()
            val_pred = pred[val_mask].cpu().numpy()
            val_recall = recall_score(val_y, val_pred, zero_division=0)
            
            logger.info(f"Epoch {epoch+1:03d}/{num_epochs}: Loss {loss.item():.4f}, Val Acc {val_acc:.4f}, Val Recall {val_recall:.4f}")
        model.train()

    # 6. Evaluation
    logger.info("Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr))
        pred = (probs > 0.5).long().squeeze()
        
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        logger.info(f"Test Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # 7. Save Model
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path / "aml_gnn.pt")
    logger.info(f"Model saved to {model_path / 'aml_gnn.pt'}")

if __name__ == "__main__":
    train()
