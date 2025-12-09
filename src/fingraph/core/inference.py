import torch
from pathlib import Path
from fingraph.core.model import AML_GraphNetwork
from fingraph.data.data_engine import TransactionLoader
from fingraph.data.download import DATA_DIR, FILE_NAME_TRANS
from fingraph.utils.logger import get_logger

logger = get_logger(__name__)

class AMLInference:
    def __init__(self, model_path: str = "models/aml_gnn.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.data = None
        self.loader = None
        self.node_list = []

    def load_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please run training first.")
        
        # Initialize model with same architecture as training
        self.model = AML_GraphNetwork(
            in_channels=16,
            hidden_channels=32,
            out_channels=1, # Updated to 1 for Binary Classification
            heads=4
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {self.model_path}")

    def load_data(self, csv_path: str = None):
        """Loads data and prepares graph for inference"""
        if csv_path is None:
            csv_path = str(DATA_DIR / FILE_NAME_TRANS)
            
        logger.info(f"Loading data from {csv_path}...")
        self.loader = TransactionLoader(csv_path)
        self.data = self.loader.export_graph_data().to(self.device)
        
        # Retrieve the node list from the loader to map indices back to account names
        if hasattr(self.loader, 'node_list'):
            self.node_list = self.loader.node_list
        else:
            # Fallback if loader doesn't have node_list (e.g. old version loaded)
            # We reconstruct it using the same query logic
            nodes_df = self.loader.conn.execute("""
                SELECT DISTINCT "From_Account" as account FROM transactions
                UNION
                SELECT DISTINCT "To_Account" as account FROM transactions
            """).fetchdf()
            self.node_list = nodes_df['account'].tolist()
            
        logger.info(f"Data loaded. Graph has {self.data.num_nodes} nodes.")

    def predict_all(self):
        """Runs inference on the entire graph"""
        if self.model is None:
            self.load_model()
        if self.data is None:
            self.load_data()
            
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            probs = torch.sigmoid(logits).squeeze() # Sigmoid for binary prob
            preds = (probs > 0.5).long()
            
        return preds, probs # Return class predictions and probability of laundering

    def get_high_risk_accounts(self, threshold=0.5, limit=100):
        """Returns a list of accounts with highest laundering probability"""
        preds, probs = self.predict_all()
        
        # Debug: Print probability stats
        logger.info(f"Probability Stats - Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")
        
        # Instead of hard threshold, get top K riskiest accounts
        # This ensures we always have something to investigate
        top_k = torch.topk(probs, k=limit)
        top_k_indices = top_k.indices
        top_k_probs = top_k.values
        
        results = []
        for i in range(len(top_k_indices)):
            idx = top_k_indices[i].item()
            prob = top_k_probs[i].item()
            
            # Only include if it has at least some non-zero risk, or if we want to see everything
            if prob > 0:
                if idx < len(self.node_list):
                    acc = self.node_list[idx]
                    results.append({"account": acc, "risk_score": prob})
        
        return results

    def predict_account(self, account_id: str):
        """Predicts risk for a specific account"""
        if self.model is None:
            self.load_model()
        if self.data is None:
            self.load_data()
            
        try:
            idx = self.node_list.index(account_id)
        except ValueError:
            return {"error": "Account not found in graph"}
            
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            prob = torch.sigmoid(logits)[idx].item()
            
        return {"account": account_id, "risk_score": prob, "is_laundering": prob > 0.5}

if __name__ == "__main__":
    # Simple test
    inference = AMLInference()
    try:
        results = inference.get_high_risk_accounts(threshold=0.9)
        print(f"Found {len(results)} high risk accounts.")
        if results:
            print("Top 5:", results[:5])
    except Exception as e:
        print(f"Inference failed (expected if model not trained yet): {e}")
