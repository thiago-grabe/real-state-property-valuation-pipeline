import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "Property Friends Basic Modeling App"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.text == "health"

def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text

def test_train():
    configuration = {
        "data_config": {"sample_key": "sample_value"},
        "pipeline_config": {"sample_key": "sample_value"}
    }

    configuration = {
  "configuration": {
        "experiment_config" : {
                      "experiment_name" : "Property_Price_Project",
                      "mlflow_tracking_url" : "HTTP://localhost:5000"
                  },
        "data_config":{
                "categorical_columns": ["type", "sector"],
                "train_cols": ["type", "sector", "net_usable_area", "net_area", "n_rooms",
                               "n_bathroom", "latitude", "longitude"],
                "target": "price"
            },
        "db_config" : {
                   "db_type": "csv",
                   "db_config" :{}
           },
        "pipeline_config":  {
              "model_module": "sklearn.ensemble",
              "model": "GradientBoostingRegressor",
              "params": {"learning_rate":0.05, "n_estimators":150, "max_depth":6, "loss":"absolute_error"}
            }
    },
  "api_key": "apikey"
},
    
    {
    "experiment_config": {
        "experiment_name": "Property_Price",
        "metric": "mse",
        "metric_goal": "min"
    }
}
    

    [
    {
        "model_name": "property_friends_best_model",
        "features": [["casa", "vitacura", 152.0, 257.0, 3, 3, -33.3794, -70.5447]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["departamento", "las condes", 140.0, 165.0, 4, 4, -33.41135, -70.56977]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["departamento", "la reina", 101.0, 101.0, 4, 3, -33.44154, -70.55704]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["departamento", "providencia", 80.0, 112.0, 1, 2, -33.42486, -70.60868]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["departamento", "vitacura", 200.0, 200.0, 3, 4, -33.4049, -70.5945]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["casa", "la reina", 406.0, 1779.0, 6, 7, -33.43718, -70.54982]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["departamento", "providencia", 66.0, 72.0, 2, 2, -33.44833, -70.62062]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["casa", "lo barnechea", 0.0, 0.0, 6, 7, -33.3356, -70.5209]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["departamento", "las condes", 74.0, 80.0, 3, 2, -33.4083, -70.5332]]
    },
    {
        "model_name": "property_friends_best_model",
        "features": [["casa", "las condes", 198.0, 310.0, 5, 2, -33.4211, -70.5522]]
    }
]

    response = client.post("/train", json=configuration)
    assert response.status_code == 200
    assert response.json() == {
        "message": "Model training initiated",
        "data_config": configuration["data_config"],
        "pipeline_config": configuration["pipeline_config"]
    }
