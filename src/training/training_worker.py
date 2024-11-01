import os
import mlflow
from mlflow.models.signature import infer_signature

from sklearn.metrics import (root_mean_squared_error, 
                             mean_absolute_percentage_error, 
                             mean_absolute_error
                            )

from src.utils.logger import get_logger
from src.data_processing import DataProcessing
from src.pipeline import TrainingPipeline
from src.database.db import DataHandler

logger = get_logger(level="INFO")

def run_training(configuration: dict):
    """
    Train a model using the provided configuration and log metrics and model artifacts using MLflow.

    Args:
        configuration (dict): A dictionary containing the configuration for the training pipeline.
            The following keys are required:
                - experiment_config (dict): Experiment configuration with keys "experiment_name" and "mlflow_tracking_url".
                - data_config (dict): Data configuration with keys "target" and "database" with keys "db_type" and "db_config".
                - pipeline_config (dict): Pipeline configuration with keys "model", "model_module", and "params".

    Returns:
        dict: A dictionary containing the run ID and experiment name.

    Raises:
        ValueError: If the configuration is invalid or if the pipeline execution fails.
    """

    experiment_name = configuration.get("experiment_config", {}).get("experiment_name")
    if not experiment_name:
        raise ValueError("Experiment name not provided in experiment_config.")

    logger.info(f"Training started for experiment: {experiment_name}")
    try:
        data_preprocessing = DataProcessing(data_config=configuration["data_config"])
        preprocessor = data_preprocessing.create_preprocessor()
        
        process = TrainingPipeline(preprocessor=preprocessor, pipeline_config=configuration["pipeline_config"])
        data_handler = DataHandler(db_type=configuration["db_config"]["db_type"],
                                   db_config=configuration["db_config"]["db_config"])
        
        train_df = data_handler.load_data(dataset_type="train")
        test_df  = data_handler.load_data(dataset_type="test")

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise ValueError("Data preparation error.")

    mlflow_tracking_url = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_url:
        raise ValueError("MLflow tracking URL not provided in experiment_config.")
    
    mlflow.set_tracking_uri(mlflow_tracking_url)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        try:
            mlflow.log_params(configuration["pipeline_config"]["params"])
            mlflow.log_param("model", configuration["pipeline_config"]["model"])
            mlflow.log_param("model_module", configuration["pipeline_config"]["model_module"])

            pipeline = process.create_pipeline()
            target_column = configuration["data_config"]["target"]
            pipeline.fit(train_df.drop(columns=[target_column], axis="columns"), train_df[target_column])

            # Model evaluation on validation data
            y_true = test_df[target_column]
            y_pred = pipeline.predict(test_df.drop(columns=[target_column], axis="columns"))
            
            # Calculate regression metrics
            mse = root_mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            
            # Log evaluation metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)
            
            logger.info(f"Logged metrics: MSE={mse}, MAE={mae}, MAPE={mape}")

            
            input_example = train_df.drop(columns=[target_column], axis="columns").head(1)
            signature = infer_signature(train_df.drop(columns=[target_column], axis="columns"), pipeline.predict(train_df.head(1)))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )
            logger.info(f"Model and metrics logged successfully for run ID: {run.info.run_id}")

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            mlflow.end_run(status="FAILED")
            raise ValueError("Training pipeline execution error.")
        
        finally:
            data_handler.close()

    logger.info(f"Training completed for experiment: {experiment_name}")

    return {"run_id": run.info.run_id, "experiment_name": experiment_name}
