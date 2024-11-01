import os
import pandas as pd
from typing import Optional

class DataHandler:
    def __init__(self, db_type: str = "csv", db_config: Optional[dict] = None):
        """
        Initialize the data handler specifically for CSV files. Other types will be passed.

        Parameters:
        - db_type: str, specifies the data source type. Supported type is:
            - "csv": Use CSV files for data storage.
        - db_config: Optional dictionary with configuration for data sources.
        
        Example Configurations:
        - CSV:
            db_config = {
                "train_path": "path/to/train.csv",
                "test_path": "path/to/test.csv"
            }
        """
        self.db_type = db_type
        self.db_config = db_config
        
        if db_type == "csv":
            # Only set paths if using CSV
            self.train_path = db_config.get("train_path", None)
            self.test_path = db_config.get("test_path", None)
        elif db_type in ["sqlite", "mysql", "postgresql"]:
            # Database setup would typically include creating a database engine.
            pass  # This is a placeholder; an engine would be set up here.
        else:
            raise ValueError("Unsupported database type. Only 'csv' is fully supported.")

    def load_data(self, dataset_type: str = "train") -> pd.DataFrame:
        """
        Load data from CSV.

        Parameters:
        - dataset_type: str, specifies which dataset to load. Either 'train' or 'test'.
        
        Returns:
        - DataFrame with the loaded data.
        """
        if self.db_type == "csv":
            # Load data from a specified CSV path

            match dataset_type:
                case "train":
                    if self.train_path is None:
                        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database",  "train.csv")
                    else:
                        path = self.train_path
                
                case "test":
                    if self.test_path is None:
                        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database",  "test.csv")
                    else:
                        path = self.test_path

                case _:
                    raise ValueError("Invalid dataset type. Only 'train' and 'test' are supported.")
                        
            data = pd.read_csv(path)
        else:
            # For databases, a query would load the table as a DataFrame
            pass  # This would include a SQL query using a database engine.

        return data

    def save_data(self, data: pd.DataFrame, dataset_type: str = "train") -> None:
        """
        Save data to CSV.

        Parameters:
        - data: DataFrame, the data to save.
        - dataset_type: str, either 'train' or 'test'.
        """
        if self.db_type == "csv":
            # Save data to the specified CSV path
            path = self.train_path if dataset_type == "train" else self.test_path
            data.to_csv(path, index=False)
        else:
            # For databases, saving would use the to_sql method
            pass  # This would use `to_sql()` to save to a database table.

    def close(self):
        """Close the database connection if needed."""
        if self.db_type in ["sqlite", "mysql", "postgresql"]:
            # Typically, dispose of the database engine connection here.
            pass  # This would call `self.engine.dispose()` for databases.
