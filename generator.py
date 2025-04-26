import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# Third-party imports
import pandas as pd
import sqlalchemy


# Local imports
from utils import logger


# Constants
DEFAULT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))

# Time interval mappings
INTERVAL_TIMEDELTA = {
    '1m': timedelta(minutes=1),
    '5m': timedelta(minutes=5),
    '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30),
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    '6h': timedelta(hours=6),
    '8h': timedelta(hours=8),
    '12h': timedelta(hours=12),
    '1d': timedelta(days=1)
}

class DatabaseConnection:
    """
    Manages SQLite database connection and data operations for trading data.
    
    Attributes:
        name (str): Database name
        interval (str): Time interval for the data
        path (str): Path to the database file
        engine: SQLAlchemy engine instance
    """
    def __init__(self, name: str, interval: str, path: str = DEFAULT_PATH):
        self.name = name
        self.interval = interval
        self.path = os.path.join(path, f"{name}.db")
        self.engine = sqlalchemy.create_engine(f'sqlite:///{self.path}')
    
    def get_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Retrieve data from the database for a given symbol and time range.
        
        Args:
            symbol: Trading pair symbol
            start: Start datetime
            end: End datetime
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            data = pd.read_sql("ohlcv_data", self.engine)
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            data = data[data['timeframe'] == self.interval]
            data = data[(data['symbol'] == symbol) & (data.index >= start) & (data.index <= end)]
            return data
        
        except Exception as e:
            logger.info(f"❌❌ Error retrieving data: {e}")
            return pd.DataFrame()

class Preprocessing:
    """Handles data preprocessing and time period calculations."""
    
    @staticmethod
    def get_adjusted_start_date(start_date: datetime, size: int, interval: str) -> datetime:
        """Calculate adjusted start date based on interval and size."""
        interval_delta = INTERVAL_TIMEDELTA[interval]
        return start_date - (interval_delta * (size-1))
    
    
    @staticmethod
    def get_periods(dt: datetime) -> pd.Series:
        """Calculate different time periods from a given datetime."""
        # Daily period
        period_1d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 12-hour period
        period_12h = dt.replace(hour=0 if dt.hour < 12 else 12, minute=0, second=0, microsecond=0)
        
        # 8-hour period
        period_8h = dt.replace(hour=(dt.hour // 8) * 8, minute=0, second=0, microsecond=0)
        
        # 4-hour period
        period_4h = dt.replace(hour=(dt.hour // 4) * 4, minute=0, second=0, microsecond=0)
        
        # 1-hour period
        period_1h = dt.replace(minute=0, second=0, microsecond=0)
        
        minute_of_hour = dt.minute
        
        # Sub-hourly periods
        period_30m = dt.replace(minute=(minute_of_hour // 30) * 30, second=0, microsecond=0)
        period_15m = dt.replace(minute=(minute_of_hour // 15) * 15, second=0, microsecond=0)
        period_5m = dt.replace(minute=(minute_of_hour // 5) * 5, second=0, microsecond=0)
        period_1m = dt.replace(second=0, microsecond=0)
        
        return pd.Series({
            '1d': period_1d, '12h': period_12h, '8h': period_8h,
            '4h': period_4h, '1h': period_1h, '30m': period_30m,
            '15m': period_15m, '5m': period_5m, '1m': period_1m
        })
    
    @staticmethod
    def execute(data: pd.DataFrame, levels: List[str], interval: str) -> pd.DataFrame:
        """
        Execute preprocessing on the data.
        
        Args:
            data: Input DataFrame
            levels: Time levels to process
            interval: Time interval
            
        Returns:
            pd.DataFrame: Processed data with multi-level index
        """
        levels_to_use = []
        for level in levels:
            levels_to_use.append(level)
            if level == interval:
                break
        
        periods_df = data.index.to_frame().iloc[:, 0].apply(Preprocessing.get_periods)
        selected_periods = [periods_df[level] for level in levels_to_use]
        
        data_multi = data.set_index(selected_periods)
        data_multi.index.names = levels_to_use
        return data_multi

class BacktestDataGenerator:
    """
    Handles data generation and management for market data from multiple sources.
    
    Attributes:
        interval (str): Time interval for the data
        size (int): Number of data points to generate
        start_date (datetime): Start date for data retrieval
        end_date (datetime): End date for data retrieval
        backtest_mode (bool): Whether running in backtest mode
    """
    
    def __init__(self, 
                 interval: str,
                 size: int = 100,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 backtest_mode: bool = True):
        
        self.levels = ['1d', '12h', '8h', '4h', '1h', '30m', '15m', '5m', '1m']
        self.size = size
        self.interval = interval
        self.start_date = Preprocessing.get_adjusted_start_date(start_date, size, interval)
        self.end_date = end_date
        self.backtest_mode = backtest_mode
        
        # Initialize data sources
        self.db = DatabaseConnection(name="database", interval=interval)
        self.data_cache = {}
        
        logger.info("DataGenerator initialized ✅")
    

    def _load_data(self, symbol: str) -> pd.DataFrame:
        
        if symbol in self.data_cache:
            return self.data_cache[symbol]
        
        try:
            data = self.db.get_data(symbol=symbol, start=self.start_date, end=self.end_date)
            logger.info(f"Data loaded from DB for {symbol}: {len(data)} bars ✅")
            data_m = Preprocessing.execute(data, self.levels, self.interval)
            self.data_cache[symbol] = data_m
            return data_m
        
        except Exception as e:
            logger.error(f"❌❌❌ Error loading from DB for {symbol}: {e}")
            raise ValueError(f"❌ No data found for {symbol}")
        

    def get_data(self, symbol: str) -> pd.DataFrame:
        """Get all data for a symbol."""
        return self._load_data(symbol)
    

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear data cache for one or all symbols.
        
        Args:
            symbol: Optional symbol to clear cache for. If None, clears all cache.
        """
        if symbol:
            self.data_cache.pop(symbol, None)
        else:
            self.data_cache.clear()
    