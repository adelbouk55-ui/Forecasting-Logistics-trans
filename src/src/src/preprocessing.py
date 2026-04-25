"""
Forecasting Model Module for Forecasting-Logistics-trans
Implements multiple forecasting models: Prophet, ARIMA, LSTM, XGBoost
"""

import logging
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    Sequential = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import configuration
from config import (
    ARIMA_ORDER, ARIMA_SEASONAL_ORDER,
    PROPHET_YEARLY_SEASONALITY, PROPHET_WEEKLY_SEASONALITY,
    PROPHET_CHANGEPOINT_PRIOR_SCALE, PROPHET_SEASONALITY_PRIOR_SCALE,
    LSTM_LOOKBACK_WINDOW, LSTM_FORECAST_HORIZON, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    XGBOOST_MAX_DEPTH, XGBOOST_LEARNING_RATE, XGBOOST_N_ESTIMATORS,
    MODELS_DIR
)

logger = logging.getLogger(__name__)

# ============================================
# BASE MODEL CLASS
# ============================================
class BaseModel:
    """Base class for all forecasting models"""
    
    def __init__(self, model_name: str):
        """Initialize base model"""
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.metrics = {}
        logger.info(f"Initialized {model_name}")
    
    def fit(self, data: pd.DataFrame) -> bool:
        """
        Fit model (to be implemented by subclasses)
        
        Args:
            data (pd.DataFrame): Training data
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError
    
    def predict(self, periods: int = 7) -> Optional[pd.DataFrame]:
        """
        Make predictions (to be implemented by subclasses)
        
        Args:
            periods (int): Number of periods to forecast
        
        Returns:
            Optional[pd.DataFrame]: Forecast DataFrame
        """
        raise NotImplementedError
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
        
        Returns:
            Dict: Evaluation metrics
        """
        try:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            r2 = r2_score(y_true, y_pred)
            
            self.metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2_score': r2
            }
            
            logger.info(f"{self.model_name} Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            return self.metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def save(self, file_path: str) -> bool:
        """
        Save model
        
        Args:
            file_path (str): Path to save model
        
        Returns:
            bool: True if successful
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, file_path: str) -> bool:
        """
        Load model
        
        Args:
            file_path (str): Path to load model
        
        Returns:
            bool: True if successful
        """
        try:
            with open(file_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_fitted = True
            logger.info(f"Model loaded: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

# ============================================
# PROPHET MODEL
# ============================================
class ProphetModel(BaseModel):
    """Facebook Prophet forecasting model"""
    
    def __init__(self):
        """Initialize Prophet model"""
        super().__init__("Prophet")
        if Prophet is None:
            logger.error("Prophet not installed. Install with: pip install prophet")
    
    def fit(self, data: pd.DataFrame, date_column: str = 'date', value_column: str = 'y') -> bool:
        """
        Fit Prophet model
        
        Args:
            data (pd.DataFrame): Training data with date and value columns
            date_column (str): Date column name
            value_column (str): Value column name
        
        Returns:
            bool: True if successful
        """
        try:
            if Prophet is None:
                logger.error("Prophet not available")
                return False
            
            # Prepare data
            df_prophet = data[[date_column, value_column]].copy()
            df_prophet.columns = ['ds', 'y']
            
            # Initialize and fit model
            self.model = Prophet(
                yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
                weekly_seasonality=PROPHET_WEEKLY_SEASONALITY,
                changepoint_prior_scale=PROPHET_CHANGEPOINT_PRIOR_SCALE,
                seasonality_prior_scale=PROPHET_SEASONALITY_PRIOR_SCALE
            )
            
            self.model.fit(df_prophet)
            self.is_fitted = True
            
            logger.info("Prophet model fitted successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            return False
    
    def predict(self, periods: int = 7) -> Optional[pd.DataFrame]:
        """
        Make predictions
        
        Args:
            periods (int): Number of periods to forecast
        
        Returns:
            Optional[pd.DataFrame]: Forecast DataFrame
        """
        try:
            if not self.is_fitted:
                logger.error("Model not fitted")
                return None
            
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            result.columns = ['date', 'forecast', 'lower', 'upper']
            
            logger.info(f"Forecast generated for {periods} periods")
            return result
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None

# ============================================
# ARIMA MODEL
# ============================================
class ARIMAModel(BaseModel):
    """ARIMA forecasting model"""
    
    def __init__(self, order: Tuple = ARIMA_ORDER):
        """
        Initialize ARIMA model
        
        Args:
            order (Tuple): (p, d, q) parameters
        """
        super().__init__("ARIMA")
        self.order = order
        if ARIMA is None:
            logger.error("ARIMA not installed. Install with: pip install statsmodels")
    
    def fit(self, data: pd.Series) -> bool:
        """
        Fit ARIMA model
        
        Args:
            data (pd.Series): Time series data
        
        Returns:
            bool: True if successful
        """
        try:
            if ARIMA is None:
                logger.error("ARIMA not available")
                return False
            
            self.model = ARIMA(data, order=self.order)
            self.model = self.model.fit()
            self.is_fitted = True
            
            logger.info(f"ARIMA{self.order} model fitted successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            return False
    
    def predict(self, periods: int = 7) -> Optional[pd.Series]:
        """
        Make predictions
        
        Args:
            periods (int): Number of periods to forecast
        
        Returns:
            Optional[pd.Series]: Forecast series
        """
        try:
            if not self.is_fitted:
                logger.error("Model not fitted")
                return None
            
            forecast = self.model.get_forecast(steps=periods)
            forecast_values = forecast.predicted_mean
            
            logger.info(f"ARIMA forecast generated for {periods} periods")
            return forecast_values
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None

# ============================================
# LSTM MODEL
# ============================================
class LSTMModel(BaseModel):
    """LSTM deep learning forecasting model"""
    
    def __init__(self, lookback: int = LSTM_LOOKBACK_WINDOW):
        """
        Initialize LSTM model
        
        Args:
            lookback (int): Lookback window size
        """
        super().__init__("LSTM")
        self.lookback = lookback
        self.scaler = None
        if Sequential is None:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM
        
        Args:
            data (np.ndarray): Input data
        
        Returns:
            Tuple: (X, y) sequences
        """
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.Series, epochs: int = LSTM_EPOCHS, batch_size: int = LSTM_BATCH_SIZE) -> bool:
        """
        Fit LSTM model
        
        Args:
            data (pd.Series): Time series data
            epochs (int): Training epochs
            batch_size (int): Batch size
        
        Returns:
            bool: True if successful
        """
        try:
            if Sequential is None:
                logger.error("LSTM not available")
                return False
            
            # Normalize data
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(data.values.reshape(-1, 1))
            
            # Create sequences
            X, y = self._create_sequences(data_scaled)
            
            # Build model
            self.model = Sequential([
                LSTM(50, activation='relu', input_shape=(self.lookback, 1)),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            self.is_fitted = True
            logger.info("LSTM model fitted successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {str(e)}")
            return False
    
    def predict(self, last_sequence: np.ndarray, periods: int = 7) -> Optional[np.ndarray]:
        """
        Make predictions
        
        Args:
            last_sequence (np.ndarray): Last sequence from data
            periods (int): Number of periods to forecast
        
        Returns:
            Optional[np.ndarray]: Forecast values
        """
        try:
            if not self.is_fitted:
                logger.error("Model not fitted")
                return None
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(periods):
                next_pred = self.model.predict(current_sequence.reshape(1, self.lookback, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                current_sequence = np.append(current_sequence[1:], next_pred)
            
            # Inverse transform
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            logger.info(f"LSTM forecast generated for {periods} periods")
            return predictions.flatten()
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None

# ============================================
# XGBOOST MODEL
# ============================================
class XGBoostModel(BaseModel):
    """XGBoost forecasting model"""
    
    def __init__(self):
        """Initialize XGBoost model"""
        super().__init__("XGBoost")
        if XGBRegressor is None:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Fit XGBoost model
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        
        Returns:
            bool: True if successful
        """
        try:
            if XGBRegressor is None:
                logger.error("XGBoost not available")
                return False
            
            self.model = XGBRegressor(
                max_depth=XGBOOST_MAX_DEPTH,
                learning_rate=XGBOOST_LEARNING_RATE,
                n_estimators=XGBOOST_N_ESTIMATORS,
                verbose=0
            )
            
            self.model.fit(X, y)
            self.is_fitted = True
            
            logger.info("XGBoost model fitted successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {str(e)}")
            return False
    
    def predict(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make predictions
        
        Args:
            X (pd.DataFrame): Feature matrix
        
        Returns:
            Optional[np.ndarray]: Predictions
        """
        try:
            if not self.is_fitted:
                logger.error("Model not fitted")
                return None
            
            predictions = self.model.predict(X)
            logger.info(f"XGBoost predictions generated for {len(X)} samples")
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """
        Get feature importance
        
        Args:
            top_n (int): Top N features
        
        Returns:
            Dict: Feature importance
        """
        try:
            if not self.is_fitted:
                logger.error("Model not fitted")
                return {}
            
            importance = self.model.get_booster().get_score(importance_type='weight')
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            return dict(sorted_importance)
        
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}

# ============================================
# ENSEMBLE MODEL
# ============================================
class EnsembleModel(BaseModel):
    """Ensemble of multiple forecasting models"""
    
    def __init__(self, models: List[BaseModel]):
        """
        Initialize ensemble
        
        Args:
            models (List[BaseModel]): List of base models
        """
        super().__init__("Ensemble")
        self.models = models
        logger.info(f"Ensemble initialized with {len(models)} models")
    
    def fit(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        Fit all models
        
        Args:
            data (pd.DataFrame): Training data
            **kwargs: Additional arguments
        
        Returns:
            bool: True if all models fitted
        """
        try:
            fitted_count = 0
            for model in self.models:
                if model.fit(data, **kwargs):
                    fitted_count += 1
            
            self.is_fitted = (fitted_count == len(self.models))
            logger.info(f"Ensemble fitted: {fitted_count}/{len(self.models)} models")
            return self.is_fitted
        
        except Exception as e:
            logger.error(f"Error fitting ensemble: {str(e)}")
            return False
    
    def predict(self, periods: int = 7, method: str = "mean") -> Optional[pd.DataFrame]:
        """
        Make ensemble predictions
        
        Args:
            periods (int): Number of periods to forecast
            method (str): Aggregation method (mean, median, weighted)
        
        Returns:
            Optional[pd.DataFrame]: Ensemble forecast
        """
        try:
            predictions = []
            
            for model in self.models:
                pred = model.predict(periods=periods)
                if pred is not None:
                    if isinstance(pred, pd.DataFrame):
                        predictions.append(pred['forecast'].values)
                    else:
                        predictions.append(pred)
            
            if not predictions:
                logger.error("No predictions from any model")
                return None
            
            predictions = np.array(predictions)
            
            if method == "mean":
                ensemble_pred = np.mean(predictions, axis=0)
            elif method == "median":
                ensemble_pred = np.median(predictions, axis=0)
            elif method == "weighted":
                weights = np.array([1.0 / len(self.models)] * len(self.models))
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
            else:
                ensemble_pred = np.mean(predictions, axis=0)
            
            result = pd.DataFrame({
                'forecast': ensemble_pred,
                'std': np.std(predictions, axis=0)
            })
            
            logger.info(f"Ensemble forecast generated using {method} method")
            return result
        
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return None

# ============================================
# MODEL FACTORY
# ============================================
class ModelFactory:
    """Factory for creating forecasting models"""
    
    @staticmethod
    def create_model(model_type: str) -> Optional[BaseModel]:
        """
        Create model by type
        
        Args:
            model_type (str): prophet, arima, lstm, xgboost
        
        Returns:
            Optional[BaseModel]: Model instance
        """
        try:
            if model_type == "prophet":
                return ProphetModel()
            elif model_type == "arima":
                return ARIMAModel()
            elif model_type == "lstm":
                return LSTMModel()
            elif model_type == "xgboost":
                return XGBoostModel()
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            return None

# ============================================
# USAGE EXAMPLES
# ============================================
if __name__ == "__main__":
    print("\n=== Forecasting Model Examples ===\n")
    
    # Generate sample data
    from src.data_loader import SampleDataGenerator
    
    df = SampleDataGenerator.generate_time_series_data(periods=100)
    
    if df is not None:
        # Example 1: Prophet
        print("1. Prophet Model")
        prophet_model = ModelFactory.create_model("prophet")
        if prophet_model and prophet_model.fit(df, date_column='date', value_column='demand'):
            forecast = prophet_model.predict(periods=7)
            if forecast is not None:
                print(forecast.head())
        
        # Example 2: ARIMA
        print("\n2. ARIMA Model")
        arima_model = ModelFactory.create_model("arima")
        if arima_model and arima_model.fit(df['demand']):
            forecast = arima_model.predict(periods=7)
            if forecast is not None:
                print(forecast)
        
        # Example 3: LSTM
        print("\n3. LSTM Model")
        lstm_model = ModelFactory.create_model("lstm")
        if lstm_model and lstm_model.fit(df['demand']):
            last_seq = df['demand'].values[-30:]
            forecast = lstm_model.predict(last_seq, periods=7)
            if forecast is not None:
                print(forecast)
        
        # Example 4: Ensemble
        print("\n4. Ensemble Model")
        models = [
            ModelFactory.create_model("prophet"),
            ModelFactory.create_model("arima")
        ]
        ensemble = EnsembleModel([m for m in models if m is not None])
        if ensemble.fit(df, date_column='date', value_column='demand'):
            forecast = ensemble.predict(periods=7, method="mean")
            if forecast is not None:
                print(forecast.head())
