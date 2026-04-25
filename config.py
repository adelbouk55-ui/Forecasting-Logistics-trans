"""
Configuration file for Forecasting-Logistics-trans application
Stores all settings, paths, and parameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# PROJECT PATHS
# ============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# APPLICATION SETTINGS
# ============================================
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

# ============================================
# FLASK API SETTINGS
# ============================================
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = DEBUG

# ============================================
# DATABASE SETTINGS (if needed)
# ============================================
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "sqlite")  # sqlite, postgresql, mysql
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///forecasting.db")
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = int(os.getenv("DATABASE_PORT", 5432))
DATABASE_USER = os.getenv("DATABASE_USER", "user")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "password")
DATABASE_NAME = os.getenv("DATABASE_NAME", "forecasting_db")

# ============================================
# DATA PROCESSING SETTINGS
# ============================================
# Data file formats
SUPPORTED_FORMATS = [".csv", ".xlsx", ".xls", ".json"]
CSV_DELIMITER = ","
EXCEL_SHEET_NAME = 0

# Data preprocessing
MISSING_VALUE_STRATEGY = "forward_fill"  # forward_fill, backward_fill, interpolate, drop
TEST_SIZE = 0.2  # Train-test split ratio
VALIDATION_SIZE = 0.1  # Validation set size
RANDOM_STATE = 42  # For reproducibility

# ============================================
# FORECASTING MODEL SETTINGS
# ============================================
# Model type: arima, prophet, lstm, xgboost, ensemble
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "prophet")

# ARIMA parameters
ARIMA_ORDER = (1, 1, 1)  # (p, d, q)
ARIMA_SEASONAL_ORDER = (1, 1, 1, 12)  # (P, D, Q, s)

# Prophet parameters
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_DAILY_SEASONALITY = False
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05
PROPHET_SEASONALITY_PRIOR_SCALE = 10.0
PROPHET_INTERVAL_WIDTH = 0.95

# LSTM parameters
LSTM_LOOKBACK_WINDOW = 30  # Days to look back
LSTM_FORECAST_HORIZON = 7  # Days to forecast
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001

# XGBoost parameters
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_N_ESTIMATORS = 100

# ============================================
# EVALUATION METRICS
# ============================================
METRICS = ["mae", "rmse", "mape", "r2_score"]

# ============================================
# LOGGING SETTINGS
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "forecasting_app.log"
LOG_MAX_BYTES = 10485760  # 10MB
LOG_BACKUP_COUNT = 5

# ============================================
# API SETTINGS
# ============================================
API_VERSION = "v1"
API_TIMEOUT = 30  # seconds
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}

# ============================================
# FORECASTING API ENDPOINTS
# ============================================
ENDPOINTS = {
    "predict": f"/api/{API_VERSION}/predict",
    "train": f"/api/{API_VERSION}/train",
    "status": f"/api/{API_VERSION}/status",
    "upload": f"/api/{API_VERSION}/upload",
    "models": f"/api/{API_VERSION}/models",
    "metrics": f"/api/{API_VERSION}/metrics",
}

# ============================================
# EXTERNAL APIs (if needed)
# ============================================
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "")

# ============================================
# CACHING SETTINGS
# ============================================
CACHE_TYPE = "simple"  # simple, redis, memcached
CACHE_TIMEOUT = 300  # 5 minutes
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ============================================
# SECURITY SETTINGS
# ============================================
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
JWT_SECRET = os.getenv("JWT_SECRET", SECRET_KEY)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1 hour

# ============================================
# NOTIFICATION SETTINGS (if needed)
# ============================================
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "False").lower() == "true"
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "noreply@forecasting.com")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# ============================================
# FEATURE FLAGS
# ============================================
ENABLE_TRAINING = True
ENABLE_PREDICTIONS = True
ENABLE_API_DOCS = True
ENABLE_CACHING = False
ENABLE_NOTIFICATIONS = False

# ============================================
# PRINT CONFIGURATION (for debugging)
# ============================================
if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
