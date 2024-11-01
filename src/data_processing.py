from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    categorical_columns: List[str]
    train_cols: List[str]
    target: str


class DataProcessing:
    def __init__(self, data_config: dict) -> None:
        
        """
        Initialize a DataProcessing object with the given data and configuration.

        Parameters
        ----------
        data_config : dict
            A dictionary containing the configuration for data processing.
            Example:
            {
                "categorical_columns": ["column1", "column2"],
                "train_cols": ["column1", "column2", "column3"],
                "target": "column3"
            }
        """

        self.config = DataConfig(**data_config)

    def create_preprocessor(self) -> ColumnTransformer:

        """
        Create a ColumnTransformer for preprocessing categorical columns.

        The ColumnTransformer encodes categorical columns using the TargetEncoder.

        Returns:
            ColumnTransformer: The preprocessor to use in a pipeline.
        """
        categorical_transformer = TargetEncoder()

        return ColumnTransformer(
            transformers=[
                ('categorical',
                categorical_transformer,
                self.config.categorical_columns)
            ]
        )

