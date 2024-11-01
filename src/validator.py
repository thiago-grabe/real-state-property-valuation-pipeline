
import os
import requests
from http import HTTPStatus
from fastapi import Body, HTTPException
from src.utils.logger import get_logger

logger = get_logger(level="INFO")

class VaultException(Exception):
    """
    Custom exception for handling Vault-related errors.

    Attributes:
        message (str): Description of the error.
        status_code (int): HTTP status code associated with the error.
    """

    def __init__(self, message: str, status_code: int):
        self.status_code = status_code
        self.message = message

    def __str__(self):
        return f"{self.status_code} : {self.message}"

class ApiKeyValidator:
    """
    A class to validate API keys against stored keys in Vault.

    Attributes:
        secret_path (str): The path to the secret in Vault.
        vault_address (str): The address of the Vault server.
        vault_token (str): The authentication token for accessing Vault.
    """

    def __init__(self):
        self.secret_path = "APIKEY"
        self.vault_address = f"{os.getenv('VAULT_ADDR')}:{os.getenv('VAULT_PORT')}"
        self.vault_token = os.getenv("VAULT_DEV_ROOT_TOKEN")

    def validate_api_key(self, user_api_key: str) -> bool:
        """
        Validates the provided API key against the stored key in Vault using HTTP requests.

        Parameters:
        - user_api_key (str): The API key provided by the user.

        Returns:
        - bool: True if the API key matches, False otherwise.
        """

        if not user_api_key:
            return False

        try:
            vault_url = f"{self.vault_address}/v1/kv/{self.secret_path}"
            headers = {"X-Vault-Token": self.vault_token}

            response = requests.get(url=vault_url, headers=headers)

            status_code, response_json = response.status_code, response.json()

            if status_code == HTTPStatus.NOT_FOUND:
                raise VaultException(
                    status_code,
                    message=f"Secret not found in {self.secret_path} in Vault"
                )

            if status_code == HTTPStatus.OK:
                if errors := response_json.get("errors"):
                    raise VaultException(
                        status_code,
                        message=f"Error retrieving API key from Vault: {errors}"
                    )

                vault_api_key = response_json.get("data", {}).get("apikey")

        except Exception as e:
            print(f"Error retrieving API key from Vault: {e}")
            return False

        return user_api_key == vault_api_key


def validate_api_key(api_key: str = Body(...)) -> None:
    """
    Validates an API key against the one stored in Vault.

    Args:
    - api_key (str): The API key provided by the user.

    Raises:
    - HTTPException: 401 if the API key is invalid.
    """
    vault_validator = ApiKeyValidator()
    if not vault_validator.validate_api_key(api_key):
        logger.warning(f"Invalid attempt to access API key: {api_key} - [UNAUTHORIZED]")
        raise HTTPException(status_code=401, detail="Invalid API key")