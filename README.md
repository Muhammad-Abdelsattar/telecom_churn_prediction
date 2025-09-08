# **Customer Churn Prediction Pipeline**

This document outlines the architecture, setup, and usage of the Customer Churn Prediction project. The project is designed as a reproducible machine learning pipeline using DVC and is served via a REST API built with FastAPI and containerized with Docker.

### **1. Project Goal & Core Technologies**

*   **Goal**: To build an end-to-end machine learning system that accurately predicts customer churn based on their account and usage data.
*   **Core Technologies**:
    *   **ML Pipeline**: DVC, Scikit-learn, XGBoost
    *   **API**: FastAPI, Uvicorn
    *   **Containerization**: Docker
    *   **Data Handling**: Pandas, OmegaConf
    *   **Programming Language**: Python 3.10

---

### **2. Repository Structure**

The repository is organized to separate concerns: the ML modeling code, the API application, the pipeline steps, and the resulting data and models.

```
├── app/                  # FastAPI application for serving predictions
│   ├── main.py           # API endpoint definitions
│   ├── inference.py      # Model loading and prediction logic
│   ├── requirements.txt  # Python dependencies for the API
│   └── schema.py         # Pydantic schema for input data validation
│
├── churn_prediction/     # Core ML code as a Python package
│   ├── data.py           # Data cleaning and preprocessing functions
│   ├── modeling.py       # Scikit-learn pipeline and model construction
│   ├── training.py       # Model training and cross-validation logic
│   ├── evaluation.py     # Model performance evaluation
│   └── utils.py          # Helper functions (e.g., saving/loading objects)
│
├── input/                # Raw and processed data (tracked by DVC)
│   ├── train.csv.dvc     # DVC pointer to the raw training data
│   ├── test.csv.dvc      # DVC pointer to the raw test data
│
├── models/               # Trained model artifacts (tracked by DVC)
│   └── pipeline.pkl      # The serialized, trained ML pipeline
│
├── reports/              # Model performance metrics
│   └── metrics/
│       ├── test.yaml     # Test set scores
│       └── validation.yaml # Cross-validation scores
│
├── steps/                # Scripts executed by the DVC pipeline stages
│   ├── prepare_data.py   # Script for the 'prepare' stage
│   ├── train.py          # Script for the 'train' stage
│   └── evaluate.py       # Script for the 'evaluate' stage
│
├── .dvc/                 # DVC internal files
├── Dockerfile            # Docker instructions for building the API image
├── dvc.yaml              # DVC pipeline definition
├── dvc.lock              # DVC lock file for reproducibility
├── params.yaml           # Configuration for model parameters
└── requirements.txt      # Python dependencies for the ML pipeline
```

---

### **3. The Machine Learning Pipeline (DVC)**

The project uses a DVC pipeline to orchestrate the ML workflow, ensuring that every step is reproducible. The pipeline is defined in `dvc.yaml`.

#### **3.1. Pipeline Stages**

1.  **`prepare`**
    *   **Purpose**: Cleans the raw data (`train.csv`, `test.csv`) by handling missing values, correcting data types, and removing uninformative columns.
    *   **Script**: `steps/prepare_data.py`
    *   **Output**: `input/prepared_train.csv`, `input/prepared_test.csv`

2.  **`train`**
    *   **Purpose**: Trains the model using the prepared training data. It builds a preprocessing and modeling pipeline, performs 5-fold cross-validation, and saves the final trained pipeline and validation metrics.
    *   **Script**: `steps/train.py`
    *   **Output**: `models/pipeline.pkl`, `reports/metrics/validation.yaml`

3.  **`evaluate`**
    *   **Purpose**: Evaluates the trained pipeline on the prepared test data to assess its performance on unseen data.
    *   **Script**: `steps/evaluate.py`
    *   **Output**: `reports/metrics/test.yaml`

#### **3.2. Configuration**

Model parameters are externalized in `params.yaml`. This allows for easy tuning without modifying the source code.

```yaml
# params.yaml
pipeline:
  filepath: models/pipeline.pkl
  model:
    use_model: xgb  # Can be switched to 'rf', 'lr', etc.
    params:
      n_estimators: 100
      random_state: 42
```

---

### **4. Prediction Service (FastAPI)**

The trained model is exposed via a REST API for easy integration with other services.

*   **Framework**: FastAPI
*   **Container**: Docker

#### **4.1. API Endpoints**

*   **`GET /`**
    *   **Description**: A simple health check endpoint.
    *   **Response**: `"Hello"`

*   **`POST /predict`**
    *   **Description**: Predicts churn for one or more customers.
    *   **Request Body**: A JSON object (for a single prediction) or a list of JSON objects (for batch predictions) matching the schema in `app/schema.py`.
    *   **Response**: A list of integer predictions (e.g., `[0, 1, 0]`, where `1` indicates churn).

---

### **5. How to Use This Project**

Follow these steps to set up the environment, reproduce the ML pipeline, and run the API.

#### **5.1. Prerequisites**

*   Python 3.10
*   Git
*   [DVC](https://dvc.org/doc/install)
*   [Docker](https://www.docker.com/get-started)

#### **5.2. Setup Instructions**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Muhammad-Abdelsattar/telecom_churn_prediction telecom_churn_prediction
    cd telecom_churn_prediction
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure DVC Remote Storage:**
    The project is configured to use Google Drive. Ensure your credentials are set up for DVC to connect to it.

4.  **Pull DVC-tracked data and models:**
    ```bash
    dvc pull
    ```
    This command downloads `train.csv`, `test.csv`, and `pipeline.pkl` from the configured Google Drive remote.

#### **5.3. Running the ML Pipeline**

To re-run the entire pipeline from data preparation to evaluation, use the `dvc repro` command. DVC will automatically execute the necessary stages.

```bash
dvc repro
```

#### **5.4. Running the Prediction API**

1.  **Build the Docker image:**
    ```bash
    docker build -t churn-prediction-api .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -d -p 8000:8000 --name churn-api churn-prediction-api
    ```
    The API is now accessible at `http://127.0.0.1:8000`.

3.  **Send a prediction request:**
    You can use tools like `curl` or access the interactive docs at `http://127.0.0.1:8000/docs`.

    **Example `curl` command:**
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes",
        "Dependents": "No", "tenure": 1, "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
      }'
    ```
