"""
Data Loader Module for Forecasting-Logistics-trans
Handles loading data from various file formats (CSV, Excel, JSON)
"""

import os
import logging
import pandas as pd
import json
from typing import Optional, Dict, List, Union
from pathlib import Path

# Import configuration
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SUPPORTED_FORMATS,
    CSV_DELIMITER, EXCEL_SHEET_NAME
)

# Import utilities
from src.utils import (
    validate_dataframe, validate_file_extension, file_exists,
    ensure_directory_exists, get_file_extension, get_dataframe_info
)

logger = logging.getLogger(__name__)

# ============================================
# CSV DATA LOADER
# ============================================
class CSVLoader:
    """Load data from CSV files"""
    
    @staticmethod
    def load(file_path: str, delimiter: str = CSV_DELIMITER) -> Optional[pd.DataFrame]:
        """
        Load CSV file
        
        Args:
            file_path (str): Path to CSV file
            delimiter (str): CSV delimiter
        
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None
        """
        try:
            if not file_exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            if validate_dataframe(df):
                logger.info(f"CSV loaded successfully: {file_path}")
                logger.info(f"Shape: {df.shape}, Columns: {list(df.columns)}")
                return df
            else:
                logger.error("DataFrame validation failed")
                return None
        
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return None
    
    @staticmethod
    def save(df: pd.DataFrame, file_path: str, delimiter: str = CSV_DELIMITER) -> bool:
        """
        Save DataFrame to CSV
        
        Args:
            df (pd.DataFrame): DataFrame to save
            file_path (str): Output file path
            delimiter (str): CSV delimiter
        
        Returns:
            bool: True if successful
        """
        try:
            ensure_directory_exists(os.path.dirname(file_path))
            df.to_csv(file_path, delimiter=delimiter, index=False)
            logger.info(f"CSV saved successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            return False

# ============================================
# EXCEL DATA LOADER
# ============================================
class ExcelLoader:
    """Load data from Excel files"""
    
    @staticmethod
    def load(file_path: str, sheet_name: Union[int, str] = EXCEL_SHEET_NAME) -> Optional[pd.DataFrame]:
        """
        Load Excel file
        
        Args:
            file_path (str): Path to Excel file
            sheet_name (Union[int, str]): Sheet name or index
        
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None
        """
        try:
            if not file_exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if validate_dataframe(df):
                logger.info(f"Excel loaded successfully: {file_path}")
                logger.info(f"Shape: {df.shape}, Columns: {list(df.columns)}")
                return df
            else:
                logger.error("DataFrame validation failed")
                return None
        
        except Exception as e:
            logger.error(f"Error loading Excel: {str(e)}")
            return None
    
    @staticmethod
    def save(df: pd.DataFrame, file_path: str, sheet_name: str = "Sheet1") -> bool:
        """
        Save DataFrame to Excel
        
        Args:
            df (pd.DataFrame): DataFrame to save
            file_path (str): Output file path
            sheet_name (str): Sheet name
        
        Returns:
            bool: True if successful
        """
        try:
            ensure_directory_exists(os.path.dirname(file_path))
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
            logger.info(f"Excel saved successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving Excel: {str(e)}")
            return False

# ============================================
# JSON DATA LOADER
# ============================================
class JSONLoader:
    """Load data from JSON files"""
    
    @staticmethod
    def load(file_path: str) -> Optional[pd.DataFrame]:
        """
        Load JSON file
        
        Args:
            file_path (str): Path to JSON file
        
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None
        """
        try:
            if not file_exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                logger.error("Unsupported JSON structure")
                return None
            
            if validate_dataframe(df):
                logger.info(f"JSON loaded successfully: {file_path}")
                logger.info(f"Shape: {df.shape}, Columns: {list(df.columns)}")
                return df
            else:
                logger.error("DataFrame validation failed")
                return None
        
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
            return None
    
    @staticmethod
    def save(df: pd.DataFrame, file_path: str, orient: str = "records") -> bool:
        """
        Save DataFrame to JSON
        
        Args:
            df (pd.DataFrame): DataFrame to save
            file_path (str): Output file path
            orient (str): JSON orientation (records, split, index, columns, values)
        
        Returns:
            bool: True if successful
        """
        try:
            ensure_directory_exists(os.path.dirname(file_path))
            df.to_json(file_path, orient=orient, indent=4)
            logger.info(f"JSON saved successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON: {str(e)}")
            return False

# ============================================
# MAIN DATA LOADER
# ============================================
class DataLoader:
    """
    Main data loader class
    Automatically detects file format and loads data
    """
    
    def __init__(self):
        """Initialize DataLoader"""
        self.loaders = {
            '.csv': CSVLoader,
            '.xlsx': ExcelLoader,
            '.xls': ExcelLoader,
            '.json': JSONLoader
        }
        logger.info("DataLoader initialized")
    
    def load(
        self, 
        file_path: str, 
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Load data from file (auto-detect format)
        
        Args:
            file_path (str): Path to data file
            **kwargs: Additional arguments for specific loaders
        
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None
        """
        try:
            # Validate file path
            if not file_exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Get file extension
            extension = get_file_extension(file_path)
            
            # Validate extension
            if not validate_file_extension(file_path, SUPPORTED_FORMATS):
                logger.error(f"Unsupported file format: {extension}")
                return None
            
            # Get appropriate loader
            loader_class = self.loaders.get(extension)
            
            if loader_class is None:
                logger.error(f"No loader found for {extension}")
                return None
            
            # Load data
            df = loader_class.load(file_path, **kwargs)
            
            if validate_dataframe(df):
                logger.info(f"Data loaded successfully: {file_path}")
                return df
            else:
                logger.error("Data validation failed")
                return None
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def load_from_directory(
        self,
        directory_path: str,
        file_pattern: str = "*"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple files from directory
        
        Args:
            directory_path (str): Path to directory
            file_pattern (str): File pattern to match
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded DataFrames
        """
        try:
            data_dict = {}
            
            if not os.path.isdir(directory_path):
                logger.error(f"Directory not found: {directory_path}")
                return data_dict
            
            # Find all matching files
            path = Path(directory_path)
            files = list(path.glob(file_pattern))
            
            logger.info(f"Found {len(files)} files in {directory_path}")
            
            # Load each file
            for file_path in files:
                if file_path.suffix in SUPPORTED_FORMATS:
                    file_name = file_path.stem
                    df = self.load(str(file_path))
                    
                    if df is not None:
                        data_dict[file_name] = df
                        logger.info(f"Loaded: {file_name}")
            
            logger.info(f"Loaded {len(data_dict)} files successfully")
            return data_dict
        
        except Exception as e:
            logger.error(f"Error loading from directory: {str(e)}")
            return {}
    
    def save(
        self,
        df: pd.DataFrame,
        file_path: str,
        format: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Save DataFrame (auto-detect format)
        
        Args:
            df (pd.DataFrame): DataFrame to save
            file_path (str): Output file path
            format (Optional[str]): File format (.csv, .xlsx, .json)
            **kwargs: Additional arguments
        
        Returns:
            bool: True if successful
        """
        try:
            if not validate_dataframe(df):
                logger.error("Invalid DataFrame")
                return False
            
            # Detect format from file path if not specified
            if format is None:
                format = get_file_extension(file_path)
            
            # Get appropriate loader
            loader_class = self.loaders.get(format)
            
            if loader_class is None:
                logger.error(f"No loader found for {format}")
                return False
            
            # Save data
            success = loader_class.save(df, file_path, **kwargs)
            
            if success:
                logger.info(f"Data saved successfully: {file_path}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False

# ============================================
# SAMPLE DATA GENERATOR
# ============================================
class SampleDataGenerator:
    """Generate sample data for testing"""
    
    @staticmethod
    def generate_time_series_data(
        periods: int = 100,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate sample time series data
        
        Args:
            periods (int): Number of time periods
            columns (List[str]): Column names
        
        Returns:
            pd.DataFrame: Sample time series data
        """
        try:
            if columns is None:
                columns = ['date', 'demand', 'price', 'inventory']
            
            import numpy as np
            
            dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
            data = {
                'date': dates,
                'demand': np.random.randint(50, 500, periods),
                'price': np.random.uniform(10, 100, periods),
                'inventory': np.random.randint(100, 1000, periods)
            }
            
            df = pd.DataFrame(data)
            logger.info(f"Generated sample data: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            return None
    
    @staticmethod
    def generate_logistics_data(
        rows: int = 1000
    ) -> pd.DataFrame:
        """
        Generate sample logistics data
        
        Args:
            rows (int): Number of rows
        
        Returns:
            pd.DataFrame: Sample logistics data
        """
        try:
            import numpy as np
            
            data = {
                'order_id': range(1, rows + 1),
                'customer_id': np.random.randint(1, 100, rows),
                'product_id': np.random.randint(1, 50, rows),
                'quantity': np.random.randint(1, 20, rows),
                'weight': np.random.uniform(0.5, 50, rows),
                'distance': np.random.uniform(10, 5000, rows),
                'delivery_time': np.random.uniform(1, 30, rows),
                'cost': np.random.uniform(10, 500, rows)
            }
            
            df = pd.DataFrame(data)
            logger.info(f"Generated logistics data: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error generating logistics data: {str(e)}")
            return None

# ============================================
# USAGE EXAMPLES
# ============================================
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()
    
    # Example 1: Load CSV file
    print("\n=== Example 1: Load CSV ===")
    df_csv = loader.load("data/raw/sample.csv")
    if df_csv is not None:
        print(df_csv.head())
    
    # Example 2: Load Excel file
    print("\n=== Example 2: Load Excel ===")
    df_excel = loader.load("data/raw/sample.xlsx")
    if df_excel is not None:
        print(df_excel.head())
    
    # Example 3: Load JSON file
    print("\n=== Example 3: Load JSON ===")
    df_json = loader.load("data/raw/sample.json")
    if df_json is not None:
        print(df_json.head())
    
    # Example 4: Generate and save sample data
    print("\n=== Example 4: Generate Sample Data ===")
    df_sample = SampleDataGenerator.generate_time_series_data(periods=30)
    if df_sample is not None:
        print(df_sample.head())
        loader.save(df_sample, "data/raw/time_series_sample.csv")
    
    # Example 5: Load multiple files from directory
    print("\n=== Example 5: Load Directory ===")
    data_dict = loader.load_from_directory("data/raw", "*.csv")
    print(f"Loaded {len(data_dict)} files")
    
    # Example 6: Get DataFrame info
    print("\n=== Example 6: DataFrame Info ===")
    if df_sample is not None:
        info = get_dataframe_info(df_sample)
        print(f"Shape: {info.get('shape')}")
        print(f"Columns: {info.get('columns')}")
        print(f"Missing values: {info.get('missing_values')}")
