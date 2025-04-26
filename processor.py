import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
from talib import MA_Type


"""
MA_Type.SMA : Simple Moving Average
MA_Type.EMA : Exponential Moving Average
MA_Type.WMA : Weighted Moving Average
MA_Type.DEMA : Double Exponential Moving Average
MA_Type.TEMA : Triple Exponential Moving Average
MA_Type.TRIMA : Triangular Moving Average
MA_Type.KAMA : Kaufman Adaptive Moving Average
MA_Type.MAMA : MESA Adaptive Moving Average
MA_Type.T3 : Triple Exponential Moving Average (T3)
"""



class Processor:
    def __init__(self, data):
        self.data = data
        
    def add_session(self):
        self.data['hour'] = self.data.index.get_level_values(-1).hour
        self.data['day_of_week'] = self.data.index.get_level_values(-1).dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6])
        
        conditions = [
            (self.data['hour'] >= 0) & (self.data['hour'] < 8),
            (self.data['hour'] >= 8) & (self.data['hour'] < 14),
            (self.data['hour'] >= 14) & (self.data['hour'] < 17),
            (self.data['hour'] >= 17) & (self.data['hour'] < 21),
            (self.data['hour'] >= 21) & (self.data['hour'] < 24)
        ]
        choices = ['Asia', 'Europe', 'US_AM', 'US_PM', 'Evening']
        self.data['session'] = np.select(conditions, choices, default='Unknown')
    
    def get_level_data(self, level):
        return self.data.groupby(level=level)['close'].last()
    
    def calculate_ema(self, level, window=20):
        price_data = self.get_level_data(level)
        return pd.Series(talib.EMA(price_data.values, timeperiod=window), index=price_data.index)
       
    def calculate_sma(self, level, window=20):
       price_data = self.get_level_data(level)
       return pd.Series(talib.SMA(price_data.values, timeperiod=window), index=price_data.index)

    def calculate_wma(self, level, window=30):
        """
        Calculate Weighted Moving Average (WMA)
        """
        price_data = self.get_level_data(level)
        return pd.Series(talib.WMA(price_data.values, timeperiod=window), index=price_data.index)

    def calculate_ma(self, level, window=30, matype=0):
        """
        Calculate Moving Average (MA) with configurable type.
        matype: 0=SMA, 1=EMA, 2=WMA, etc. (see TA-Lib docs)
        """
        price_data = self.get_level_data(level)
        return pd.Series(talib.MA(price_data.values, timeperiod=window, matype=matype), index=price_data.index)

    
    def calculate_macd(self, level, fastperiod=12, slowperiod=26, signalperiod=9, 
                     fastmatype=MA_Type.EMA, slowmatype=MA_Type.EMA, signalmatype=MA_Type.EMA):
        """
        Calculate MACD with controllable MA types using MACDEXT
        By default uses EMA for all components
        Available MA types:
        - MA_Type.SMA : Simple Moving Average
        - MA_Type.EMA : Exponential Moving Average
        - MA_Type.WMA : Weighted Moving Average
        - MA_Type.DEMA : Double Exponential Moving Average
        - MA_Type.TEMA : Triple Exponential Moving Average
        - MA_Type.TRIMA : Triangular Moving Average
        - MA_Type.KAMA : Kaufman Adaptive Moving Average
        - MA_Type.MAMA : MESA Adaptive Moving Average
        - MA_Type.T3 : Triple Exponential Moving Average (T3)
        """
        price_data = self.get_level_data(level)
        macd, signal, hist = talib.MACDEXT(
            price_data.values,
            fastperiod=fastperiod,
            fastmatype=fastmatype,
            slowperiod=slowperiod,
            slowmatype=slowmatype,
            signalperiod=signalperiod,
            signalmatype=signalmatype
        )
        return (
            pd.Series(macd, index=price_data.index),
            pd.Series(signal, index=price_data.index),
            pd.Series(hist, index=price_data.index)
        )

    def calculate_rsi(self, level, window=14):
       price_data = self.get_level_data(level)
       return pd.Series(talib.RSI(price_data.values, timeperiod=window), index=price_data.index)

    def calculate_kama(self, level, window=30):
        price_data = self.get_level_data(level)
        return pd.Series(talib.KAMA(price_data.values, timeperiod=window), index=price_data.index)

    def calculate_bollinger(self, level, window=20, nbdevup=2, nbdevdn=2, matype=MA_Type.EMA):
        """
        Calculate Bollinger Bands with controllable MA type
        Available MA types:
        - MA_Type.SMA : Simple Moving Average
        - MA_Type.EMA : Exponential Moving Average
        - MA_Type.WMA : Weighted Moving Average
        - MA_Type.DEMA : Double Exponential Moving Average
        - MA_Type.TEMA : Triple Exponential Moving Average
        - MA_Type.TRIMA : Triangular Moving Average
        - MA_Type.KAMA : Kaufman Adaptive Moving Average
        - MA_Type.MAMA : MESA Adaptive Moving Average
        - MA_Type.T3 : Triple Exponential Moving Average (T3)
        """
        price_data = self.get_level_data(level)
        upper, middle, lower = talib.BBANDS(price_data.values, timeperiod=window,
                                            nbdevup=nbdevup,
                                            nbdevdn=nbdevdn,
                                            matype=matype)
        return (
            pd.Series(upper, index=price_data.index),
            pd.Series(middle, index=price_data.index),
            pd.Series(lower, index=price_data.index)
        )

    def calculate_atr(self, level, window=14):
       price_data = self.data.groupby(level=level).agg({
           'high': 'max',
           'low': 'min',
           'close': 'last'
       })
       return pd.Series(
            talib.ATR(price_data['high'].values, price_data['low'].values, price_data['close'].values, timeperiod=window),
            index=price_data.index)

    def calculate_adx(self, level, window=14):
        price_data = self.data.groupby(level=level).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        return pd.Series(
            talib.ADX(
                price_data['high'].values,
                price_data['low'].values,
                price_data['close'].values,
                timeperiod=window
            ),
            index=price_data.index
        )

    def calculate_sar(self, level, acceleration=0.02, maximum=0.2):
        price_data = self.data.groupby(level=level).agg({
            'high': 'max',
            'low': 'min'
        })
        return pd.Series(
            talib.SAR(
                price_data['high'].values,
                price_data['low'].values,
                acceleration=acceleration,
                maximum=maximum
            ),
            index=price_data.index
        )

    def calculate_stochrsi(self, level, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        price_data = self.get_level_data(level)
        fastk, fastd = talib.STOCHRSI(
            price_data.values,
            timeperiod=timeperiod,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
            fastd_matype=fastd_matype
        )
        return (
            pd.Series(fastk, index=price_data.index),
            pd.Series(fastd, index=price_data.index)
        )

    def calculate_ppo(self, level, fastperiod=12, slowperiod=26, matype=0):
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.PPO(price_data.values,
                      fastperiod=fastperiod, slowperiod=slowperiod,
                      matype=matype), index=price_data.index
        )

    def calculate_donchian(self, level, window=20):
        """
        Calculate Donchian Channels
        Returns upper, middle and lower bands
        Upper band = highest high over N periods
        Lower band = lowest low over N periods
        Middle band = (upper + lower) / 2
        """
        price_data = self.data.groupby(level=level).agg({
            'high': 'max',
            'low': 'min'
        })
        close = self.get_level_data(level)
        
        # Calculate rolling highest high and lowest low
        upper = price_data['high'].rolling(window=window).max()
        lower = price_data['low'].rolling(window=window).min()
        middle = (upper + lower) / 2
        
        return (
            pd.Series(upper, index=price_data.index),
            pd.Series(middle, index=price_data.index),
            pd.Series(lower, index=price_data.index)
        )

    def calculate_sar(self, level, acceleration=0, maximum=0):
        price_data = self.data.groupby(level=level).agg({
            'high': 'max',
            'low': 'min'
        })
        return pd.Series(talib.SAR(price_data['high'], price_data['low'], acceleration=acceleration, maximum=maximum), index=price_data.index)
    
    def calculate_sar_ext(self, level,
                        startvalue=0.0, offsetonreverse=0.0,
                        accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
                        accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2):
        """
        Calculate Parabolic SAR Extended (SAREXT)
        """
        price_data = self.data.groupby(level=level).agg({
            'high': 'max',
            'low': 'min'
        })
        return pd.Series(
            talib.SAREXT(
                price_data['high'].values,
                price_data['low'].values,
                startvalue=startvalue,
                offsetonreverse=offsetonreverse,
                accelerationinitlong=accelerationinitlong,
                accelerationlong=accelerationlong,
                accelerationmaxlong=accelerationmaxlong,
                accelerationinitshort=accelerationinitshort,
                accelerationshort=accelerationshort,
                accelerationmaxshort=accelerationmaxshort
            ),
            index=price_data.index
        ).apply(lambda x: abs(x))

    def calculate_mama(self, level, fastlimit=0.5, slowlimit=0.05):
        """
        Calculate MESA Adaptive Moving Average (MAMA) and FAMA using TA-Lib.
        Returns two series: MAMA and FAMA.
        """
        price_data = self.get_level_data(level)
        mama, fama = talib.MAMA(price_data.values, fastlimit=fastlimit, slowlimit=slowlimit)
        return pd.Series(mama, index=price_data.index), pd.Series(fama, index=price_data.index)

    def apply_mama(self, name, level, fastlimit=0.5, slowlimit=0.05):
        """
        Apply MESA Adaptive Moving Average (MAMA) and FAMA to the dataframe.
        """
        mama, fama = self.calculate_mama(level, fastlimit, slowlimit)
        self.data[f'{name}_mama_{level}'] = self.data.index.get_level_values(level).map(mama)
        self.data[f'{name}_fama_{level}'] = self.data.index.get_level_values(level).map(fama)

    # Application methods
    def apply_ema(self, name, level, window=20):
        ema = self.calculate_ema(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(ema)
    
    def apply_sma(self, name, level, window=20):
        sma = self.calculate_sma(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(sma)

    def apply_macd(self, name, level, fastperiod=12, slowperiod=26, signalperiod=9,
                  fastmatype=MA_Type.EMA, slowmatype=MA_Type.EMA, signalmatype=MA_Type.EMA):
        """
        Apply MACD with controllable MA types to the dataframe
        By default uses EMA for all components
        """
        macd_line, signal_line, histogram = self.calculate_macd(
            level, fastperiod, slowperiod, signalperiod,
            fastmatype, slowmatype, signalmatype
        )
        self.data[f'{name}_line_{level}'] = self.data.index.get_level_values(level).map(macd_line)
        self.data[f'{name}_signal_{level}'] = self.data.index.get_level_values(level).map(signal_line)
        self.data[f'{name}_histogram_{level}'] = self.data.index.get_level_values(level).map(histogram)

    def apply_rsi(self, name, level, window=14):
        rsi = self.calculate_rsi(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(rsi)

    def apply_kama(self, name, level, window=30):
        kama = self.calculate_kama(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(kama)

    def apply_bollinger(self, name, level, window=20, nbdevup=2, nbdevdn=2, matype=MA_Type.EMA):
        """
        Apply Bollinger Bands with controllable MA type to the dataframe
        By default uses EMA
        """
        upper, middle, lower = self.calculate_bollinger(level, window, nbdevup, nbdevdn, matype)
        self.data[f'{name}_upper_{level}'] = self.data.index.get_level_values(level).map(upper)
        self.data[f'{name}_middle_{level}'] = self.data.index.get_level_values(level).map(middle)
        self.data[f'{name}_lower_{level}'] = self.data.index.get_level_values(level).map(lower)
    
    def apply_atr(self, name, level, window=14):
        atr = self.calculate_atr(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(atr)
    
    def apply_adx(self, name, level, window=14):
        adx = self.calculate_adx(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(adx)
        
    
    def apply_sar(self, name, level, acceleration=0.02, maximum=0.2):
        sar = self.calculate_sar(level, acceleration, maximum)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(sar)
        
    
    def apply_stochrsi(self, name, level, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        fastk, fastd = self.calculate_stochrsi(level, timeperiod, fastk_period, fastd_period, fastd_matype)
        self.data[f'{name}_k_{level}'] = self.data.index.get_level_values(level).map(fastk)
        self.data[f'{name}_d_{level}'] = self.data.index.get_level_values(level).map(fastd)

    def apply_ppo(self, name, level, fastperiod=12, slowperiod=26, matype=0):
        ppo = self.calculate_ppo(level, fastperiod, slowperiod, matype)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(ppo)

    def apply_donchian(self, name, level, window=20):
        """
        Apply Donchian Channels to the dataframe
        """
        upper, middle, lower = self.calculate_donchian(level, window)
        self.data[f'{name}_upper_{level}'] = self.data.index.get_level_values(level).map(upper)
        self.data[f'{name}_middle_{level}'] = self.data.index.get_level_values(level).map(middle)
        self.data[f'{name}_lower_{level}'] = self.data.index.get_level_values(level).map(lower)

    def apply_ma(self, name, level, window=30, matype=0):
        ma = self.calculate_ma(level, window, matype)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(ma)

    def apply_wma(self, name, level, window=30):
        wma = self.calculate_wma(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(wma)

    def apply_sar_ext(self, name, level,
                  startvalue=0.02, offsetonreverse=0.0,
                  accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
                  accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2):
        sar = self.calculate_sar_ext(level,
            startvalue, offsetonreverse,
            accelerationinitlong, accelerationlong, accelerationmaxlong,
            accelerationinitshort, accelerationshort, accelerationmaxshort
        )
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(sar)
