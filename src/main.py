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
    """
    Trains a model with the provided configuration.

    Args:
    - configuration (dict): A dictionary containing the configuration for the training pipeline.
        The following keys are required:
            - experiment_config (dict): A dictionary containing the experiment configuration with the following keys:
                - experiment_name (str): The name of the experiment.
            - data_config (dict): A dictionary containing the data configuration with the following keys:
                - target (str): The name of the target column.
                - database (dict): A dictionary containing the database configuration with the following keys:
                    - db_type (str): The type of the database. Supported types are: "csv".
                    - db_config (dict): A dictionary containing the database configuration specific to the database type.
            - pipeline_config (dict): A dictionary containing the pipeline configuration with the following keys:
                - model (str): The name of the model class.
                - model_module (str): The module name of the model class.
                - params (dict): A dictionary containing the hyperparameters of the model.

    Returns:
    - dict: A dictionary containing the run ID and experiment name.

    Raises:
    - HTTPException: 400 if the configuration is invalid.

    Example 1:
    {
        "experiment_config": {
            "experiment_name": "Property-Friends-Experiment"
        },
        "data_config": {
            "categorical_columns": ["type", "sector"],
            "train_cols": [
                "type",
                "sector",
                "net_usable_area",
                "net_area",
                "n_rooms",
                "n_bathroom",
                "latitude",
                "longitude"
            ],
            "target": "price"
        },
        "db_config": {
            "db_type": "csv",
            "db_config": {}
        },
        "pipeline_config": {
            "model_module": "sklearn.ensemble",
            "model": "GradientBoostingRegressor",
            "params": {
                "learning_rate": 0.05,
                "n_estimators": 150,
                "max_depth": 6,
                "loss": "absolute_error"
            }
        }
    }

    Example 2:
    {
        "experiment_config": {
            "experiment_name": "Property_Price_Project"
        },
        "data_config": {
            "categorical_columns": ["type", "sector"],
            "train_cols": [
                "type",
                "sector",
                "net_usable_area",
                "net_area",
                "n_rooms",
                "n_bathroom",
                "latitude",
                "longitude"
            ],
            "target": "price"
        },
        "db_config": {
            "db_type": "csv",
            "db_config": {}
        },
        "pipeline_config": {
            "model_module": "sklearn.linear_model",
            "model": "LinearRegression",
            "params": {
                "fit_intercept": True
            }
        }
    }

    Example 3:
    {
        "experiment_config": {
            "experiment_name": "Property_Price_Project"
        },
        "data_config": {
            "categorical_columns": ["type", "sector"],
            "train_cols": [
                "type",
                "sector",
                "net_usable_area",
                "net_area",
                "n_rooms",
                "n_bathroom",
                "latitude",
                "longitude"
            ],
            "target": "price"
        },
        "db_config": {
            "db_type": "csv",
            "db_config": {}
        },
        "pipeline_config": {
            "model_module": "sklearn.ensemble",
            "model": "RandomForestRegressor",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        }
    }
    """

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
    """
    Retrieve the status, result, and error information of a specific job.

    Args:
        job_id (str): The unique identifier of the job to be fetched.

    Returns:
        dict: A dictionary containing the following keys:
            - job_id (str): The unique identifier of the job.
            - status (str): The current status of the job (e.g., queued, started, finished, failed).
            - result (any): The result of the job if it has finished successfully, otherwise None.
            - error (str): The error information if the job has failed, otherwise None.

    Raises:
        HTTPException: If the job with the specified job_id is not found, returns a 404 status code.
    """
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
    Retrieve the best model from an MLflow experiment in the MLflow Model Registry.

    Args:
        configuration (dict): A dictionary containing the configuration for the best model retrieval.
            The following keys are required:
                - experiment_config (dict): Experiment configuration with the following keys:
                    - experiment_name (str): The name of the experiment to retrieve the best model from.
                    - metric (str): The metric to use for selecting the best model. One of "mse", "mae", "mape".
                    - metric_goal (str): The direction of the metric for selecting the best model. One of "min", "max".

    Returns:
        dict: A dictionary containing the following keys:
            - experiment_name (str): The name of the experiment.
            - best_run_id (str): The ID of the best run in the experiment.
            - best_metric_value (float): The value of the metric for the best run. One of "mse", "mae", "mape".
            - model_uri (str): The URI of the best model artifact in the MLflow Model Registry.

    Raises:
        HTTPException: If the experiment or run is not found, or if there is an error retrieving the best model.

    Example:
        {
            "experiment_config": {
                "experiment_name": "Property-Friends-Experiment",
                "metric": "mse",
                "metric_goal": "min"
            }
            
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
    Register the best model of an experiment as a production model based on a metric.

    Args:
        configuration (dict): A dictionary containing the configuration for the best model retrieval.
            The following keys are required:
                - experiment_config (dict): Experiment configuration with the following keys:
                    - experiment_name (str): The name of the experiment to retrieve the best model from.
                    - model_name (str): The name of the model to register.
                    - metric (str): The metric to use for selecting the best model. One of "mse", "mae", "mape".
                    - metric_goal (str): The direction of the metric for selecting the best model. One of "min", "max".

    Returns:
        dict: A dictionary containing the following keys:
            - experiment_name (str): The name of the experiment.
            - best_run_id (str): The ID of the best run in the experiment.
            - best_metric_value (float): The value of the metric for the best run. One of "mse", "mae", "mape".
            - model_uri (str): The URI of the best model artifact in the MLflow Model Registry.

    Raises:
        HTTPException: If the experiment or run is not found, or if there is an error retrieving the best model.

    Example:
        {
            "experiment_config": {
                "experiment_name": "Property-Friends-Experiment",
                "model_name": "Property-Friends-Model",
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
    Single Predict using the registered model.

    Args:
        input_data (dict): A dictionary containing the following keys:
            - model_name (str): The name of the model to use for prediction.
            - features (dict): A dictionary containing the features to use for prediction.
                - dataframe_split (dict): A dictionary containing the split data.
                    - columns (list): A list of column names.
                    - data (list): A list of lists containing the data.

    Returns:
        JSONResponse: A JSON response containing the prediction.

    Raises:
        HTTPException: If the model name is not provided or if the features are not provided.
        HTTPException: If the prediction fails.

        
     Example 01:
        {
            "model_name": "property_friends_best_model",
            "features": {
                "type": ["departamento"],
                "sector": ["vitacura"],
                "net_usable_area": [140.0],
                "net_area": [170.0],
                "n_rooms": [4.0],
                "n_bathroom": [4.0],
                "latitude": [-33.40123],
                "longitude": [-70.58056]
            }
        }

    Example 02:
        {
            "model_name": "property_friends_best_model",
            "features": [
                {
                    "type": "casa",
                    "sector": "lo barnechea",
                    "net_usable_area": 300.0,
                    "net_area": 500.0,
                    "n_rooms": 6.0,
                    "n_bathroom": 5.0,
                    "latitude": -33.3561,
                    "longitude": -70.5072
                },
                {
                    "type": "departamento",
                    "sector": "las condes",
                    "net_usable_area": 120.0,
                    "net_area": 150.0,
                    "n_rooms": 3.0,
                    "n_bathroom": 2.0,
                    "latitude": -33.411,
                    "longitude": -70.5697
                },
                {
                    "type": "casa",
                    "sector": "la reina",
                    "net_usable_area": 250.0,
                    "net_area": 400.0,
                    "n_rooms": 5.0,
                    "n_bathroom": 3.0,
                    "latitude": -33.4375,
                    "longitude": -70.5499
                },
                {
                    "type": "departamento",
                    "sector": "providencia",
                    "net_usable_area": 80.0,
                    "net_area": 120.0,
                    "n_rooms": 2.0,
                    "n_bathroom": 2.0,
                    "latitude": -33.425,
                    "longitude": -70.609
                },
                {
                    "type": "casa",
                    "sector": "vitacura",
                    "net_usable_area": 200.0,
                    "net_area": 300.0,
                    "n_rooms": 4.0,
                    "n_bathroom": 3.0,
                    "latitude": -33.4049,
                    "longitude": -70.5945
                }
            ]
        }
    """

    model_name = input_data.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required in the input.")
    
    if not input_data.get("features"):
        raise HTTPException(status_code=400, detail="Features are required in the input for prediction.")
    
    model_features = [dict(zip(input_data.get("features")[i].keys(), input_data.get("features")[i].values())) for i in range(len(input_data.get("features")))]

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
