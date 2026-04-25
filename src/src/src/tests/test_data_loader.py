"""
Unit tests for Data Loader Module
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

from src.data_loader import (
    CSVLoader, ExcelLoader, JSONLoader, DataLoader, SampleDataGenerator
)

logger = None

class TestCSVLoader(unittest.TestCase):
    """Test CSV Loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': np.random.randint(1, 100, 10),
            'price': np.random.uniform(10, 100, 10)
        })
    
    def test_save_csv(self):
        """Test saving CSV file"""
        file_path = os.path.join(self.temp_dir, 'test.csv')
        result = CSVLoader.save(self.sample_data, file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(file_path))
    
    def test_load_csv(self):
        """Test loading CSV file"""
        file_path = os.path.join(self.temp_dir, 'test.csv')
        CSVLoader.save(self.sample_data, file_path)
        df = CSVLoader.load(file_path)
        
        self.assertIsNotNone(df)
        self.assertEqual(df.shape, self.sample_data.shape)
    
    def test_load_nonexistent_csv(self):
        """Test loading non-existent CSV"""
        df = CSVLoader.load('/nonexistent/file.csv')
        self.assertIsNone(df)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestExcelLoader(unittest.TestCase):
    """Test Excel Loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': np.random.randint(1, 100, 10)
        })
    
    def test_save_excel(self):
        """Test saving Excel file"""
        file_path = os.path.join(self.temp_dir, 'test.xlsx')
        result = ExcelLoader.save(self.sample_data, file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(file_path))
    
    def test_load_excel(self):
        """Test loading Excel file"""
        file_path = os.path.join(self.temp_dir, 'test.xlsx')
        ExcelLoader.save(self.sample_data, file_path)
        df = ExcelLoader.load(file_path)
        
        self.assertIsNotNone(df)
        self.assertEqual(df.shape[0], self.sample_data.shape[0])
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestJSONLoader(unittest.TestCase):
    """Test JSON Loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
    
    def test_save_json(self):
        """Test saving JSON file"""
        file_path = os.path.join(self.temp_dir, 'test.json')
        result = JSONLoader.save(self.sample_data, file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(file_path))
    
    def test_load_json(self):
        """Test loading JSON file"""
        file_path = os.path.join(self.temp_dir, 'test.json')
        JSONLoader.save(self.sample_data, file_path)
        df = JSONLoader.load(file_path)
        
        self.assertIsNotNone(df)
        self.assertEqual(df.shape[0], self.sample_data.shape[0])
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestDataLoader(unittest.TestCase):
    """Test main Data Loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'demand': np.random.randint(50, 500, 10),
            'price': np.random.uniform(10, 100, 10)
        })
    
    def test_load_csv(self):
        """Test loading CSV with DataLoader"""
        file_path = os.path.join(self.temp_dir, 'test.csv')
        self.sample_data.to_csv(file_path, index=False)
        df = self.loader.load(file_path)
        
        self.assertIsNotNone(df)
        self.assertEqual(df.shape[0], 10)
    
    def test_load_unsupported_format(self):
        """Test loading unsupported format"""
        file_path = os.path.join(self.temp_dir, 'test.txt')
        with open(file_path, 'w') as f:
            f.write('test')
        
        df = self.loader.load(file_path)
        self.assertIsNone(df)
    
    def test_save_multiple_formats(self):
        """Test saving in multiple formats"""
        formats = ['csv', 'xlsx', 'json']
        
        for fmt in formats:
            file_path = os.path.join(self.temp_dir, f'test.{fmt}')
            result = self.loader.save(self.sample_data, file_path)
            self.assertTrue(result)
            self.assertTrue(os.path.exists(file_path))
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestSampleDataGenerator(unittest.TestCase):
    """Test Sample Data Generator"""
    
    def test_generate_time_series(self):
        """Test generating time series data"""
        df = SampleDataGenerator.generate_time_series_data(periods=50)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 50)
        self.assertEqual(df.shape[1], 4)
    
    def test_generate_logistics_data(self):
        """Test generating logistics data"""
        df = SampleDataGenerator.generate_logistics_data(rows=100)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 100)
        self.assertIn('order_id', df.columns)
        self.assertIn('cost', df.columns)
    
    def test_data_quality(self):
        """Test data quality"""
        df = SampleDataGenerator.generate_time_series_data(periods=100)
        
        self.assertFalse(df.isnull().any().any())
        self.assertTrue(len(df) > 0)

if __name__ == '__main__':
    unittest.main()
