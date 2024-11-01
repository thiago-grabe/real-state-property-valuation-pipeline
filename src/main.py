import os
from fastapi import FastAPI, Body, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from redis import Redis
from rq import Queue
from rq.job import Job

import pandas as pd

from src.utils.logger import get_logger
from src.validator import validate_api_key
from src.training.training_worker import run_training
from src.utils.config import ExperimentConfig

import mlflow
import mlflow.sklearn

logger = get_logger(level="INFO")
app = FastAPI(title="Property-Friends-Basic-Modeling")

# Redis configuration from environment variables
redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = os.getenv("REDIS_PORT", "6379")
redis_password = os.getenv("REDIS_PASSWORD", "redis")
redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"

# Connect to Redis
redis_conn = Redis.from_url(redis_url)
queue = Queue("training-queue", connection=redis_conn)


@app.get("/", response_class=PlainTextResponse)
async def home():
    return "Property Friends Basic Modeling App"


@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "health"


@app.post("/train", response_class=JSONResponse, dependencies=[Depends(validate_api_key)])
def train(configuration: dict = Body(...)):
    logger.info("Training job being added to the queue...")
    # job = queue.enqueue("src.training.training_worker.run_training", configuration)
    job = queue.enqueue(run_training, configuration)
    logger.info(f"Training job enqueued with ID: {job.id}")
    return {
        "message": "Training job enqueued",
        "job_id": job.id
    }


@app.get("/job_status/{job_id}", response_class=JSONResponse)
def get_job_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job.get_status()
    result = job.result if status == "finished" else None
    error = job.exc_info if job.is_failed else None

    return {
        "job_id": job_id,
        "status": status,
        "result": result,
        "error": error
    }


@app.post("/best_model", response_class=JSONResponse, dependencies=[Depends(validate_api_key)])
async def get_best_model(configuration: dict = Body(...)):
    """
    Retrieves the best model from an MLflow experiment based on a specified metric.

    {
  "configuration": {"experiment_config": {
    "experiment_name": "eval",
    "model_name": "property_friends_best_model",
    "metric": "mse",
    "metric_goal": "min"
}},
  "api_key": "apikey"
}

    """
    try:
        experiment_config = ExperimentConfig(**configuration.get("experiment_config", {}))
        experiment_name = experiment_config.experiment_name
        metric_name = experiment_config.metric
        metric_goal = experiment_config.metric_goal
    except Exception as e:
        error_details = {
            "error": "Invalid experiment configuration",
            "details": str(e)
        }
        return JSONResponse(content=error_details, status_code=400)
    
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # Ensure the MLflow tracking URI is set
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found.")
        
        experiment_id = experiment.experiment_id
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if metric_goal == 'min' else 'DESC'}"]
        )
        
        if runs.empty:
            raise HTTPException(status_code=404, detail="No runs found in experiment.")
        
        best_run = runs.iloc[0]  # The first run after sorting is the best based on the metric
        best_run_id = best_run.run_id
        best_metric_value = best_run[f"metrics.{metric_name}"]
        
        # Fetch model artifacts for the best run
        model_uri = f"runs:/{best_run_id}/model"
        model_details = {
            "experiment_name": experiment_name,
            "best_run_id": best_run_id,
            "best_metric_value": best_metric_value,
            "model_uri": model_uri
        }
        
        logger.info(f"Best model fetched with Run ID: {best_run_id}, {metric_name}: {best_metric_value}")
        
        return JSONResponse(content=model_details)

    except Exception as e:
        logger.error(f"Error fetching best model: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve best model.")
    

@app.post("/deploy", response_class=JSONResponse, dependencies=[Depends(validate_api_key)])
async def register_best_model(configuration: dict = Body(...)):
    """
    Deploy the best model from an MLflow experiment in the MLflow Model Registry.

        {
        "experiment_config": {
            "experiment_name": "Property-Friends-Experiment",
            "model_name": "property_friends_best_model",
            "metric": "mse",
            "metric_goal": "min"
        }
    }

    """
    
    try:
        experiment_config = ExperimentConfig(**configuration.get("experiment_config", {}))
        experiment_name = experiment_config.experiment_name
        model_name = experiment_config.model_name
        metric_name = experiment_config.metric
        metric_goal = experiment_config.metric_goal
    except Exception as e:
        error_details = {
            "error": "Invalid experiment configuration",
            "details": str(e)
        }
        return JSONResponse(content=error_details, status_code=400)

    if not all([experiment_name, model_name, metric_name]):
        raise HTTPException(status_code=400, detail="Missing required configuration parameters.")

    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found.")

        experiment_id = experiment.experiment_id
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if metric_goal == 'min' else 'DESC'}"]
        )

        if runs.empty:
            raise HTTPException(status_code=404, detail="No runs found in experiment.")

        best_run = runs.iloc[0]
        best_run_id = best_run.run_id

        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name, tags={"stage": "PROD"})

        logger.info(f"Best model registered with Run ID: {best_run_id}, model name: {model_name}")
        return JSONResponse(content={"message": f"Model registered as '{model_name}'", "run_id": best_run_id})

    except Exception as e:
        logger.error(f"Error registering best model: {e}")
        raise HTTPException(status_code=500, detail="Failed to register best model.")


@app.post("/predict", response_class=JSONResponse, dependencies=[Depends(validate_api_key)])
async def predict(input_data: dict = Body(...)):
    """
    Predict using the latest registered model.
    """
    model_name = input_data.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required in the input.")
    
    if not input_data.get("features"):
        raise HTTPException(status_code=400, detail="Features are required in the input for prediction.")
    
    model_features = {key: [value] for key, value in input_data.get("features").items()}

    features = pd.DataFrame.from_dict(model_features)

    try:
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.pyfunc.load_model(model_uri)

        prediction = model.predict(features)
        
        logger.info(f"Prediction completed with model '{model_name}': {prediction}")
        return JSONResponse(content={"prediction": prediction.tolist()})

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
