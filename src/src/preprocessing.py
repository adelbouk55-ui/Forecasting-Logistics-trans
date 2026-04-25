"""
Data Preprocessing Module for Forecasting-Logistics-trans
Handles data cleaning, transformation, and feature engineering
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Import configuration
from config import (
    MISSING_VALUE_STRATEGY, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE
)

# Import utilities
from src.utils import (
    handle_missing_values, remove_outliers, remove_duplicates,
    normalize_dataframe, validate_dataframe
)

logger = logging.getLogger(__name__)

# ============================================
# DATA CLEANER
# ============================================
class DataCleaner:
    """Clean and prepare data"""
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df (pd.DataFrame): Input DataFrame
            subset (Optional[List[str]]): Columns to check for duplicates
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        try:
            initial_count = len(df)
            df_clean = remove_duplicates(df, subset=subset)
            removed_count = initial_count - len(df_clean)
            logger.info(f"Removed {removed_count} duplicate rows")
            return df_clean
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}")
            return df
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = MISSING_VALUE_STRATEGY,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            df (pd.DataFrame): Input DataFrame
            strategy (str): forward_fill, backward_fill, drop, mean, median
            threshold (float): Drop columns with missing > threshold
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        try:
            initial_missing = df.isnull().sum().sum()
            
            # Drop columns with too many missing values
            df = df.dropna(axis=1, thresh=len(df) * (1 - threshold))
            
            # Handle remaining missing values
            if strategy == "forward_fill":
                df = df.fillna(method='ffill').fillna(method='bfill')
            elif strategy == "backward_fill":
                df = df.fillna(method='bfill').fillna(method='ffill')
            elif strategy == "drop":
                df = df.dropna()
            elif strategy == "mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == "median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            final_missing = df.isnull().sum().sum()
            logger.info(f"Missing values: {initial_missing} -> {final_missing}")
            return df
        
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[List[str]]): Columns to check
            method (str): iqr or zscore
            threshold (float): Outlier threshold
        
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
            
            initial_count = len(df)
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                df = remove_outliers(df, col, method=method, threshold=threshold)
            
            removed_count = initial_count - len(df)
            logger.info(f"Removed {removed_count} outlier rows ({method})")
            return df
        
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            return df
    
    @staticmethod
    def fix_data_types(df: pd.DataFrame, type_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """
        Fix data types
        
        Args:
            df (pd.DataFrame): Input DataFrame
            type_mapping (Optional[Dict]): Column -> dtype mapping
        
        Returns:
            pd.DataFrame: DataFrame with corrected types
        """
        try:
            if type_mapping:
                for col, dtype in type_mapping.items():
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(dtype)
                        except Exception as e:
                            logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
            
            logger.info("Data types fixed")
            return df
        
        except Exception as e:
            logger.error(f"Error fixing data types: {str(e)}")
            return df

# ============================================
# FEATURE ENGINEERING
# ============================================
class FeatureEngineer:
    """Create and engineer features"""
    
    @staticmethod
    def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create date/time features
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_column (str): Date column name
        
        Returns:
            pd.DataFrame: DataFrame with date features
        """
        try:
            if date_column not in df.columns:
                logger.warning(f"Date column {date_column} not found")
                return df
            
            df[date_column] = pd.to_datetime(df[date_column])
            
            df['year'] = df[date_column].dt.year
            df['month'] = df[date_column].dt.month
            df['day'] = df[date_column].dt.day
            df['dayofweek'] = df[date_column].dt.dayofweek
            df['quarter'] = df[date_column].dt.quarter
            df['week'] = df[date_column].dt.isocalendar().week
            df['is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
            
            logger.info(f"Created date features from {date_column}")
            return df
        
        except Exception as e:
            logger.error(f"Error creating date features: {str(e)}")
            return df
    
    @staticmethod
    def create_lag_features(
        df: pd.DataFrame,
        column: str,
        lags: List[int] = [1, 7, 30]
    ) -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to create lags for
            lags (List[int]): Lag periods
        
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        try:
            if column not in df.columns:
                logger.warning(f"Column {column} not found")
                return df
            
            for lag in lags:
                df[f'{column}_lag_{lag}'] = df[column].shift(lag)
            
            logger.info(f"Created lag features for {column}")
            return df
        
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            return df
    
    @staticmethod
    def create_rolling_features(
        df: pd.DataFrame,
        column: str,
        windows: List[int] = [7, 30]
    ) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to create rolling features for
            windows (List[int]): Window sizes
        
        Returns:
            pd.DataFrame: DataFrame with rolling features
        """
        try:
            if column not in df.columns:
                logger.warning(f"Column {column} not found")
                return df
            
            for window in windows:
                df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
                df[f'{column}_rolling_std_{window}'] = df[column].rolling(window).std()
                df[f'{column}_rolling_min_{window}'] = df[column].rolling(window).min()
                df[f'{column}_rolling_max_{window}'] = df[column].rolling(window).max()
            
            logger.info(f"Created rolling features for {column}")
            return df
        
        except Exception as e:
            logger.error(f"Error creating rolling features: {str(e)}")
            return df
    
    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame,
        columns: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[Tuple[str, str]]): Pairs of columns to multiply
        
        Returns:
            pd.DataFrame: DataFrame with interaction features
        """
        try:
            for col1, col2 in columns:
                if col1 in df.columns and col2 in df.columns:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            logger.info(f"Created {len(columns)} interaction features")
            return df
        
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
            return df

# ============================================
# DATA SCALER
# ============================================
class DataScaler:
    """Scale and normalize data"""
    
    def __init__(self, method: str = "standard"):
        """
        Initialize scaler
        
        Args:
            method (str): standard or minmax
        """
        self.method = method
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        logger.info(f"Scaler initialized: {method}")
    
    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Fit scaler
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[List[str]]): Columns to fit
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
            
            self.scaler.fit(df[columns])
            self.columns = columns
            logger.info(f"Scaler fitted on {len(columns)} columns")
        
        except Exception as e:
            logger.error(f"Error fitting scaler: {str(e)}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: Scaled DataFrame
        """
        try:
            df_scaled = df.copy()
            df_scaled[self.columns] = self.scaler.transform(df[self.columns])
            logger.info(f"Data transformed using {self.method} scaler")
            return df_scaled
        
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            return df
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform data
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[List[str]]): Columns to fit and transform
        
        Returns:
            pd.DataFrame: Scaled DataFrame
        """
        self.fit(df, columns)
        return self.transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform (scale back to original)
        
        Args:
            df (pd.DataFrame): Scaled DataFrame
        
        Returns:
            pd.DataFrame: Original scale DataFrame
        """
        try:
            df_original = df.copy()
            df_original[self.columns] = self.scaler.inverse_transform(df[self.columns])
            logger.info("Data inverse transformed")
            return df_original
        
        except Exception as e:
            logger.error(f"Error inverse transforming: {str(e)}")
            return df

# ============================================
# DATA SPLITTER
# ============================================
class DataSplitter:
    """Split data for training and testing"""
    
    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        shuffle: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """
        Split data into train and test
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (Optional[str]): Target column for stratification
            test_size (float): Test set size (0-1)
            random_state (int): Random seed
            shuffle (bool): Shuffle data
        
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            if target_column and target_column in df.columns:
                y = df[target_column]
                X = df.drop(columns=[target_column])
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state,
                    shuffle=shuffle
                )
                
                logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
                return X_train, X_test, y_train, y_test
            else:
                X_train, X_test = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=random_state,
                    shuffle=shuffle
                )
                
                logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
                return X_train, X_test, None, None
        
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return None, None, None, None
    
    @staticmethod
    def train_validation_test_split(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        test_size: float = TEST_SIZE,
        validation_size: float = VALIDATION_SIZE,
        random_state: int = RANDOM_STATE
    ) -> Tuple:
        """
        Split data into train, validation, and test
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (Optional[str]): Target column
            test_size (float): Test set size
            validation_size (float): Validation set size
            random_state (int): Random seed
        
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            if target_column and target_column in df.columns:
                y = df[target_column]
                X = df.drop(columns=[target_column])
                
                # First split: train + val vs test
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state
                )
                
                # Second split: train vs validation
                val_ratio = validation_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val,
                    test_size=val_ratio,
                    random_state=random_state
                )
                
                logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                X_train_val, X_test = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=random_state
                )
                
                val_ratio = validation_size / (1 - test_size)
                X_train, X_val = train_test_split(
                    X_train_val,
                    test_size=val_ratio,
                    random_state=random_state
                )
                
                logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
                return X_train, X_val, X_test, None, None, None
        
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return None, None, None, None, None, None
    
    @staticmethod
    def time_series_split(
        df: pd.DataFrame,
        date_column: str,
        test_size: float = TEST_SIZE,
        validation_size: float = VALIDATION_SIZE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_column (str): Date column name
            test_size (float): Test set size
            validation_size (float): Validation set size
        
        Returns:
            Tuple: (train_df, val_df, test_df)
        """
        try:
            df = df.sort_values(date_column)
            n = len(df)
            
            test_start = int(n * (1 - test_size - validation_size))
            val_start = int(n * (1 - test_size))
            
            train_df = df[:test_start]
            val_df = df[test_start:val_start]
            test_df = df[val_start:]
            
            logger.info(f"Time series split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df
        
        except Exception as e:
            logger.error(f"Error splitting time series: {str(e)}")
            return None, None, None

# ============================================
# MAIN PREPROCESSOR
# ============================================
class DataPreprocessor:
    """Main data preprocessing pipeline"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.scaler = None
        logger.info("DataPreprocessor initialized")
    
    def preprocess(
        self,
        df: pd.DataFrame,
        remove_duplicates_flag: bool = True,
        handle_missing_flag: bool = True,
        remove_outliers_flag: bool = True,
        scale_flag: bool = False,
        scaling_method: str = "standard"
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Input DataFrame
            remove_duplicates_flag (bool): Remove duplicates
            handle_missing_flag (bool): Handle missing values
            remove_outliers_flag (bool): Remove outliers
            scale_flag (bool): Scale data
            scaling_method (str): Scaling method
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        try:
            logger.info(f"Starting preprocessing pipeline: {df.shape}")
            
            # Remove duplicates
            if remove_duplicates_flag:
                df = self.cleaner.remove_duplicates(df)
            
            # Handle missing values
            if handle_missing_flag:
                df = self.cleaner.handle_missing_values(df)
            
            # Remove outliers
            if remove_outliers_flag:
                df = self.cleaner.remove_outliers(df)
            
            # Scale data
            if scale_flag:
                self.scaler = DataScaler(method=scaling_method)
                df = self.scaler.fit_transform(df)
            
            logger.info(f"Preprocessing complete: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            return df

# ============================================
# USAGE EXAMPLES
# ============================================
if __name__ == "__main__":
    print("\n=== Data Preprocessing Examples ===\n")
    
    # Create sample data
    from src.data_loader import SampleDataGenerator
    
    df = SampleDataGenerator.generate_time_series_data(periods=100)
    
    if df is not None:
        # Example 1: Clean data
        print("1. Cleaning data...")
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(df)
        print(f"Shape: {df_clean.shape}")
        
        # Example 2: Engineer features
        print("\n2. Engineering features...")
        engineer = FeatureEngineer()
        df_features = engineer.create_date_features(df_clean, 'date')
        df_features = engineer.create_lag_features(df_features, 'demand', lags=[1, 7])
        print(f"Columns: {list(df_features.columns)}")
        
        # Example 3: Scale data
        print("\n3. Scaling data...")
        scaler = DataScaler(method="standard")
        df_scaled = scaler.fit_transform(df_features)
        print(f"Scaled shape: {df_scaled.shape}")
        
        # Example 4: Split data
        print("\n4. Splitting data...")
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.train_test_split(
            df_scaled,
            target_column='demand'
        )
        
        if X_train is not None:
            print(f"Train shape: {X_train.shape}")
            print(f"Test shape: {X_test.shape}")
        
        # Example 5: Complete preprocessing
        print("\n5. Complete preprocessing...")
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess(
            df,
            remove_duplicates_flag=True,
            handle_missing_flag=True,
            remove_outliers_flag=False,
            scale_flag=True
        )
        print(f"Processed shape: {df_processed.shape}")
