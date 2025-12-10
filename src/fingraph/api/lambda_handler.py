import json
import logging
import random
from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError
from fingraph.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Pydantic Models for Validation ---
class TransactionEvent(BaseModel):
    transaction_id: str
    from_account: str
    to_account: str
    amount: float = Field(..., gt=0, description="Transaction amount must be positive")
    timestamp: str
    currency: str = "USD"

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_suspicious: bool
    explanation: str

# --- Mock Model Loader ---
class ModelService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.model = cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        logger.info("Loading AML Graph Network model...")
        try:
            from fingraph.core.inference import AMLInference
            # We initialize the inference engine but don't load data yet to save cold start time
            # In a real Lambda, we might load a small subgraph from DynamoDB instead of the full CSV
            self.inference = AMLInference(model_path="/var/task/models/aml_gnn.pt")
            # For this demo, we assume the model file exists in the container
            return self.inference
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def predict(self, transaction: TransactionEvent) -> float:
        logger.info(f"Running inference for transaction {transaction.transaction_id}")
        
        # In a real serverless architecture (Lambda), we cannot load the entire 500k node graph into memory.
        # Instead, we would fetch the "Ego Graph" (neighbors) of the involved accounts from a graph DB (Neptune/DynamoDB).
        
        # For this MVP demonstration, we will use a simplified heuristic if the full graph isn't loaded,
        # or use the actual model if we can (but likely too heavy for Lambda cold start).
        
        # Mock logic for Lambda demo (since we don't have a live GraphDB connection here):
        # 1. High amounts near structuring limit ($9000-$9999)
        if 9000 <= transaction.amount < 10000:
            return 0.85 # High risk (Structuring)
            
        # 2. Very high amounts
        if transaction.amount > 50000:
            return 0.65 # Medium risk
            
        return 0.02 # Low risk
            base_prob += 0.4
        if transaction.amount > 50000:
            base_prob += 0.3
            
        # Add some noise
        prob = min(base_prob + random.uniform(0, 0.1), 1.0)
        return prob

# --- Lambda Handler ---
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entry point for real-time fraud detection.
    """
    logger.info("Received event")
    
    try:
        # 1. Parse and Validate Input
        # AWS API Gateway often wraps the body in a string
        if "body" in event:
            body = json.loads(event["body"])
        else:
            body = event
            
        transaction = TransactionEvent(**body)
        
        # 2. Get Model and Predict
        model_service = ModelService()
        probability = model_service.predict(transaction)
        
        # 3. Construct Response
        is_suspicious = probability > 0.5
        explanation = "High value transaction detected." if is_suspicious else "Transaction appears normal."
        
        response_data = PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(probability, 4),
            is_suspicious=is_suspicious,
            explanation=explanation
        )
        
        logger.info(f"Prediction complete: {response_data.model_dump_json()}")
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": response_data.model_dump_json()
        }

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid request format", "details": e.errors()})
        }
    except Exception as e:
        logger.error(f"Internal error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"})
        }
