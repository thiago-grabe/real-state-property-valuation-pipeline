from dataclasses import dataclass
from importlib import import_module

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@dataclass
class PipelineConfig:
    model_module: str
    model: str
    params: dict


class TrainingPipeline():

    """
    A Pipeline object is used to create a scikit-learn pipeline for regression tasks.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A ColumnTransformer object that encodes categorical columns and scales numerical columns.
    pipeline_config : PipelineConfig
        A configuration object that specifies the model and its hyperparameters.

    Examples
    --------
    >>> pipeline_config1 = {
    ...     "model_module": "sklearn.linear_model",
    ...     "model": "LinearRegression",
    ...     "params": {"fit_intercept": True}
    ... }
    >>> pipeline_config2 = {
    ...     "model_module": "sklearn.ensemble",
    ...     "model": "GradientBoostingRegressor",
    ...     "params": {"n_estimators": 50, "max_depth": 5, "learning_rate": 0.1}
    ... }
    >>> pipeline_config3 = {
    ...     "model_module": "sklearn.ensemble",
    ...     "model": "GradientBoostingRegressor",
    ...     "params": {"learning_rate":0.01, "n_estimators":300, "max_depth":5, "loss":"absolute_error"}
    ... }
    >>> pipeline_config4 = {
    ...     "model_module": "sklearn.svm",
    ...     "model": "SVR",
    ...     "params": {"kernel": "rbf", "C": 10, "epsilon": 0.1}
    ... }
    """

    def __init__(self, preprocessor: ColumnTransformer, pipeline_config: dict) -> Pipeline:

        """
        Initialize the pipeline with a preprocessor and a model configuration.

        Parameters
        ----------
        preprocessor : ColumnTransformer
            A ColumnTransformer object that encodes categorical columns and scales numerical columns.
        pipeline_config : dict
            A configuration dictionary that specifies the model and its hyperparameters.

        Returns
        -------
        None
        """
        self.preprocessor = preprocessor

        self.pipeline_config = PipelineConfig(**pipeline_config)

        model_module = import_module(self.pipeline_config.model_module)
        model_class = getattr(model_module, self.pipeline_config.model)

        self.model = model_class(**self.pipeline_config.params)

    def create_pipeline(self) -> Pipeline:

        """
        Creates a scikit-learn Pipeline instance with the preprocessor and model specified
        in the pipeline configuration.

        Returns
        -------
        pipeline : Pipeline
            A Pipeline instance that can be used for training and prediction.
        """
        return Pipeline(
            steps=[
                ('preprocessor', self.preprocessor),
                ('model', self.model)
            ]
        )

