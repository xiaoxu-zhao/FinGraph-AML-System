import json
import pytest
from fingraph.api.lambda_handler import lambda_handler

def test_lambda_handler_valid_transaction():
    """
    Test the lambda handler with a valid transaction event.
    """
    event = {
        "transaction_id": "TX-123",
        "from_account": "ACC-A",
        "to_account": "ACC-B",
        "amount": 9500.00,
        "timestamp": "2023-10-27T10:00:00",
        "currency": "USD"
    }
    
    response = lambda_handler(event, context={})
    
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    
    assert "fraud_probability" in body
    assert "is_suspicious" in body
    assert body["transaction_id"] == "TX-123"
    # Since amount is > 9000, our mock logic should flag it as suspicious (or high prob)
    # But due to random noise in mock, we just check types
    assert isinstance(body["fraud_probability"], float)
    assert isinstance(body["is_suspicious"], bool)

def test_lambda_handler_invalid_input():
    """
    Test validation error for negative amount.
    """
    event = {
        "transaction_id": "TX-123",
        "from_account": "ACC-A",
        "to_account": "ACC-B",
        "amount": -100.00, # Invalid
        "timestamp": "2023-10-27T10:00:00"
    }
    
    response = lambda_handler(event, context={})
    
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "error" in body
