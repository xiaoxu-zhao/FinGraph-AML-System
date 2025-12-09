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
        # In a real scenario, we would load the PyTorch model here:
        # model = AML_GraphNetwork(...)
        # model.load_state_dict(torch.load("model.pt"))
        return "MockModel"

    def predict(self, transaction: TransactionEvent) -> float:
        # Simulate inference latency
        # In reality, we would construct a subgraph around the transaction and pass to GNN
        logger.info(f"Running inference for transaction {transaction.transaction_id}")
        
        # Mock logic: High amounts are more suspicious for this demo
        base_prob = 0.01
        if transaction.amount > 9000:
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
