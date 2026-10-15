from fastapi.testclient import TestClient 
from app.main import app 

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == "healthy"
    assert "lstm_model_loaded" in data
    assert "tokenizer_loader" in data

def test_predict_positive_review():
    payload = {"text": "This dress is absolutely stunning and fits like a glove!"}
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "orginal_review" in data 
        assert "processed_review" in data 
        assert "sentiment" in data
        assert "review_topics" in data 
        assert isinstance(data["confidence_percent"], float)
    else:
        assert response.status_code == 500 


def test_predict_validation_error():
    response = client.post("/predict", json={"review": "Bad clothing item"})
    assert response.status_code == 422