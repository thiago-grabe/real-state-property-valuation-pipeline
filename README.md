# Property-Friends Real Estate Model Deployment

## Overview

This project is designed to productivize a model that predicts residential property values in Chile. Using Docker, FastAPI, Redis, and MLflow, the solution provides a scalable, deployable API for real estate valuation based on given property features. The project also supports running in a production environment with essential security and logging mechanisms.

## Contents
1. [Project Structure](#project-structure)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Environment Variables](#environment-variables)
5. [Usage](#usage)
6. [API Documentation](#api-documentation)
7. [Assumptions & Future Improvements](#assumptions--future-improvements)

---

## Project Structure

The project is organized as follows:

```
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── src
│   ├── main.py                      # FastAPI main entry point
│   ├── data_processing.py           # Data processing functions
│   ├── pipeline.py                  # Pipeline orchestration
│   ├── training
│   │   ├── training_worker.py       # Asynchronous training worker
│   │   └── __init__.py
│   ├── utils
│   │   ├── config.py                # Configuration management
│   │   ├── logger.py                # Logger setup for API and training
│   │   └── __init__.py
│   ├── database
│   │   ├── db.py                    # Database connection management
│   │   ├── train.csv                # Training data
│   │   └── test.csv                 # Test data
│   └── validator.py                 # API key validation
├── Dockerfile                       # Dockerfile for API server and worker
├── docker-compose.yaml              # Docker Compose file
└── .env                             # Environment variables 
```

**Notes:**
- `train.csv` and `test.csv` for this project are hidden due to privacy concerns. They should be available at the project repository under the `database` directory.

## Technologies Used

- **FastAPI**: For creating RESTful API endpoints.
- **MLflow**: Experiment tracking, model registry, and versioning.
- **Redis & Redis Queue**: Message queue for asynchronous tasks.
- **Vault**: Secure storage for API keys.
- **Docker Compose**: Containerized environment for deployment.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/property-friends-real-estate
    cd property-friends-real-estate
    ```

2. **Set Environment Variables**:
   Configure a `.env` file in the root of the project with the following variables (details in [Environment Variables](#environment-variables)):

   ```env
    # MLflow configuration
    MLFLOW_PORT=5000
    MLFLOW_TRACKING_URI="HTTP://tracking_server:5000"


    # Redis Configuration
    REDIS_HOST=redis
    REDIS_PORT=6379
    REDIS_PASSWORD=redis
    REDIS_SECRET_KEY=redis_secret_key
    REDIS_AUTH_TOKEN=redis
    REDIS_UI_PORT=8001

    # Vault Configuration
    VAULT_ADDR = "http://vault"
    VAULT_PORT=8200
    VAULT_DEV_ROOT_TOKEN=vault_token

    # API Key to access ML models
    API_KEY=any-api-key-you-want
   ```

3. **Build and Start the Containers**:
   Ensure Docker is running, then execute:

   ```bash
   docker-compose up --build
   ```

   The containers should initialize for the following services:
   - **MLflow tracking server**
   - **Vault for API key management**
   - **Redis and Redis Queue**
   - **FastAPI for serving predictions and training tasks**

4. **Verify Setup**:
   Once containers are up, check the logs or access:
   - FastAPI at `http://localhost:8000`
   - MLflow tracking server at `http://localhost:5000`
   - Vault at `http://localhost:8200`

## Environment Variables

Set the following environment variables for configuration in a `.env` file:

```env
#### MLflow Configuration
MLFLOW_PORT=5000                       # The port for the MLflow tracking server
MLFLOW_TRACKING_URI="http://tracking_server:5000"  # The URI for the MLflow tracking server

#### Redis Configuration
REDIS_HOST=redis                       # Hostname for the Redis server
REDIS_PORT=6379                        # Port for the Redis server
REDIS_PASSWORD=redis                   # Password to access the Redis server
REDIS_SECRET_KEY=redis_secret_key      # Secret key for Redis
REDIS_AUTH_TOKEN=redis                 # Authentication token for Redis
REDIS_UI_PORT=8001                     # Port for the Redis UI

#### Vault Configuration
VAULT_ADDR="http://vault"              # Address for the Vault server
VAULT_PORT=8200                        # Port for the Vault server
VAULT_DEV_ROOT_TOKEN=vault_token       # Root token for Vault development

#### API Key to Access ML Models
API_KEY=any-api-key-you-want           # The API key used to access the ML models

```

## Usage

1. **Health Check**: Confirm the service is running:

   ```bash
   curl http://localhost:8000/health
   ```

2. **Training a Model**:
   Submit a POST request to `/train` with the model configuration. Example:

   ```json
   POST http://localhost:8000/train
   {
     "experiment_config": {
         "experiment_name": "Property-Friends-Experiment"
     },
     "data_config": {
         "categorical_columns": ["type", "sector"],
         "train_cols": ["type", "sector", "net_usable_area", "net_area", "n_rooms", "n_bathroom", "latitude", "longitude"],
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
   ```

3. **Check Training Job Status**:
   Retrieve the status of a training job:

   ```bash
   curl http://localhost:8000/job_status/{job_id}
   ```

4. **Fetch Best Model**:
   Retrieve the best model based on a specified metric (e.g., `mse`):

   ```json
   POST http://localhost:8000/best_model
   {
     "experiment_config": {
         "experiment_name": "Property-Friends-Experiment",
         "metric": "mse",
         "metric_goal": "min"
     }
   }
   ```

5. **Make Predictions**:
   Request predictions from the deployed model:

   ```json
   POST http://localhost:8000/predict
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
         }
     ]
   }
   ```

## API Documentation

The FastAPI instance auto-generates API documentation, available at:
- **Swagger**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These documents provide endpoints, parameters, and example requests for each API route.

---

### Assumptions

- **Input Data Structure**: The model assumes that input data aligns with the schema used in training (e.g., `type`, `sector`, `net_usable_area`, etc.). For robust predictions, input data should be validated to match the column names and data types used during model training.
- **Data Source Abstraction**: Currently, the model is trained and evaluated on static CSV files (`train.csv` and `test.csv`). These files are meant for initial testing and evaluation only, not for production. In future deployments, this setup can be adapted to pull data directly from a database or a cloud data storage, abstracting the data source to use database connections (like PostgreSQL or MySQL) or cloud storage (like AWS S3). Implementing this abstraction would enable the model to handle live data updates and make it more flexible for production use.

### Immediate Next Steps for ML Pipeline Improvements
To enhance the robustness and flexibility of the ML pipeline, consider implementing the following steps:
1. **Dynamic Train-Test Split**: Integrate a train-test split directly into the pipeline. This allows for random data shuffling and partitioning, which improves model generalization and performance monitoring.
   
   Example:
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['price']), data['price'], test_size=0.2, random_state=42)
   ```

2. **Expanded Model Evaluation Metrics**: In addition to mean squared error (MSE), include other metrics like mean absolute error (MAE), mean absolute percentage error (MAPE), and R-squared. These provide a more comprehensive evaluation of model performance.
   
   Example:
   ```python
   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

   mse = mean_squared_error(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   ```

3. **Automated Cross-Validation**: Integrate cross-validation to validate the model's stability. Libraries like `sklearn.model_selection` provide tools to automate cross-validation, improving accuracy and robustness.
   
   Example:
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
   ```

4. **Feature Engineering Pipeline**: Expand the pipeline to include feature engineering steps, such as one-hot encoding for categorical variables, scaling for continuous variables, and automated feature selection. This step would improve the model's predictive power and ensure a more robust feature set.

---

### Future Improvements

1. **Database Error Handling**:
   - **Objective**: Improve robustness and reliability when integrating a database for real-time data retrieval.
   - **Implementation**:
     - Add detailed error logging to capture and report specific database errors (e.g., connection timeouts, authentication failures, malformed queries).
     - Include data validation and format checks to ensure the retrieved data matches the expected schema, preventing data quality issues in the pipeline.
     - Implement other types of databases as requested.

2. **API Key Security Enhancements**:
   - **Objective**: Strengthen security beyond basic API keys to protect sensitive data and limit access to authorized users only.
   - **Implementation**:
     - Replace or supplement Vault-based API Key management with OAuth 2.0 or JWT, providing tokens that expire and require periodic reauthorization.
     - Implement role-based access control (RBAC) to restrict specific API endpoints or model versions to particular users or groups.
     - Use HTTPS to encrypt communication between clients and the API, securing transmitted data.
   - **Example**: Use JWT with token expiration, allowing each user session to be secure and refreshed periodically.

3. **Continuous Model Monitoring and Retraining**:
   - **Objective**: Maintain model accuracy and relevance by monitoring performance and retraining based on defined thresholds.
   - **Implementation**:
     - Deploy real-time monitoring tools to track model performance metrics such as prediction error, latency, and input data distribution.
     - Configure alerts to notify the team if error metrics exceed thresholds or if significant data drift is detected, indicating the model may need retraining.
     - Automate the retraining pipeline to trigger based on monitoring results or scheduled intervals. The retraining process should include evaluation on test data and automated deployment if performance improves.
   - **Example**: Set an alert to trigger if the model’s MSE exceeds a 10% increase over the last month. This alert initiates a retraining process using the latest dataset, ensuring model quality remains high.

4. **Containerization Improvements**:
   - **Objective**: Optimize the container deployment process for performance, security, and scalability.
   - **Implementation**:
     - Use Docker multi-stage builds to create slim, production-ready images. This approach reduces the image size by excluding development dependencies and unnecessary files.
     - Add security layers by scanning images for vulnerabilities, using tools like Clair or Trivy.
     - Use orchestration tools like Kubernetes for auto-scaling, load balancing, and easy rollback, which allow for seamless scaling and management of model instances.
     - Configure Kubernetes health checks (liveness and readiness probes) to monitor container health and ensure only healthy instances serve traffic.
   - **Example**: A Docker multi-stage build could create a small production image from a base Python image, removing development dependencies like `pytest`. Kubernetes would manage multiple instances, scaling up based on load and replacing failed containers automatically.

5. **Advanced Logging and Auditing for Compliance**:
   - **Objective**: Enable end-to-end traceability of predictions for compliance, debugging, and analytics.
   - **Implementation**:
     - Implement structured logging to record every prediction request with associated metadata, including timestamps, model version, input data, and prediction result.
     - Store logs in a centralized location (e.g., ELK stack or a cloud-based logging service) to facilitate easy searching, filtering, and analysis.
     - Create audit trails for API access and user interactions with the model, meeting regulatory compliance standards (e.g., GDPR).
   - **Example**: Log all API calls to a centralized system, tagging them with unique identifiers to trace each prediction back to its input data and model version.

6. **Improved Data Pipeline and Feature Engineering**:
   - **Objective**: Enhance the model’s predictive power by improving data preprocessing and feature selection.
   - **Implementation**:
     - Expand the data pipeline to include feature engineering steps such as handling missing values, encoding categorical variables, scaling numerical features, and generating new features based on existing data.
     - Use feature selection techniques like Recursive Feature Elimination (RFE) or mutual information to reduce noise and improve model efficiency.
     - Implement automated data validation checks to verify data quality before training and inference, alerting the team if issues are found.
   - **Example**: Apply one-hot encoding to categorical variables and scale numerical features using StandardScaler. RFE could identify the top features that contribute to model performance, enhancing prediction accuracy and reducing computation time.

7. **Automated Model Staging with MLflow**:
   - **Objective**: Simplify model management by automating model staging and transitioning models through development, staging, and production phases.
   - **Implementation**:
     - Use MLflow’s model registry to automatically register new models with metadata (e.g., performance metrics, creation date, author).
     - Establish an approval process where models in the “Staging” phase are reviewed, validated, and then promoted to “Production” if they meet performance criteria.
     - Set up automated testing in staging to run A/B testing or canary deployment before promoting the model.
   - **Example**: When a new model version is trained, it’s automatically registered in MLflow. The model is then tested in staging, and if it outperforms the current production model, it’s promoted to production.

8. **API Gateway with Rate Limiting and Caching**:
   - **Objective**: Control API usage, reduce response time, and handle high traffic effectively.
   - **Implementation**:
     - Deploy an API gateway to manage API requests, implement rate limiting to prevent abuse, and cache responses for common queries to improve performance.
     - Configure the gateway to handle authentication, logging, and request validation before forwarding to the model server.
   - **Example**: AWS API Gateway or Kong could enforce a rate limit of 1000 requests per user per day and cache the most recent 10 predictions for quick retrieval on repeated requests.

9. **Error Handling and Graceful Degradation**:
    - **Objective**: Ensure the system remains functional and provides helpful feedback, even when issues arise.
    - **Implementation**:
      - Implement exception handling across the pipeline to manage errors like database timeouts, API failures, and model loading issues.
      - Set up a fallback mechanism, such as using a simpler model or returning a predefined response when the main model is unavailable.
    - **Example**: If the model service is down, return a cached prediction or a placeholder response with an error message. Log the issue for future analysis.

---