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



class Indicator:
    def __init__(self, data):
        self.data = data
        self._cache = {}
        #self.key = str(data.index.get_level_values(-1)[-1])
    
    def _hash_data(self):
        return hash(pd.util.hash_pandas_object(self.data.iloc[[-1]]).sum())
    
    def get_level_data(self, level: str) -> pd.DataFrame:
        #key = (level, self._hash_data())
        level_ = level.replace('m', 'min')
        key = level_
        if key not in self._cache:
            self._cache[key] = self.data.groupby(level=level).agg({
                f'open_{level_}': 'first',
                f'close_{level_}': 'first',
                f'high_{level_}': 'first',
                f'low_{level_}': 'first',
                f'volume_{level_}': 'first'
            }).rename(columns={
                f'open_{level_}': 'open',
                f'close_{level_}': 'close',
                f'high_{level_}': 'high',
                f'low_{level_}': 'low',
                f'volume_{level_}': 'volume'
            })
        return self._cache[key]
    
    def add_session(self):
        self.data['hour'] = self.data.index.get_level_values(-1).hour
        self.data['day_of_week'] = self.data.index.get_level_values(-1).dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6])
        self.data['half'] = self.data['hour'] <= 12
        
        conditions = [
            (self.data['hour'] >= 0) & (self.data['hour'] < 8),
            (self.data['hour'] >= 8) & (self.data['hour'] < 14),
            (self.data['hour'] >= 14) & (self.data['hour'] < 17),
            (self.data['hour'] >= 17) & (self.data['hour'] < 21),
            (self.data['hour'] >= 21) & (self.data['hour'] < 24)
        ]
        choices = ['Asia', 'Europe', 'US_AM', 'US_PM', 'Evening']
        self.data['session'] = np.select(conditions, choices, default='Unknown')
    
    def calculate_ema(self, level, window=20):
        price_data = self.get_level_data(level)
        return pd.Series(talib.EMA(price_data['close'].values, timeperiod=window), index=price_data.index)
       
    def calculate_sma(self, level, window=20):
       price_data = self.get_level_data(level)
       return pd.Series(talib.SMA(price_data['close'].values, timeperiod=window), index=price_data.index)
   
    def calculate_mama(self, level, fastlimit=0.5, slowlimit=0.05):
        """
        Calculate MESA Adaptive Moving Average (MAMA) and FAMA using TA-Lib.
        Returns two series: MAMA and FAMA.
        """
        price_data = self.get_level_data(level)
        mama, fama = talib.MAMA(price_data['close'].values, fastlimit=fastlimit, slowlimit=slowlimit)
        return pd.Series(mama, index=price_data.index), pd.Series(fama, index=price_data.index)

    def calculate_wma(self, level, window=30):
        """
        Calculate Weighted Moving Average (WMA)
        """
        price_data = self.get_level_data(level)
        return pd.Series(talib.WMA(price_data['close'].values, timeperiod=window), index=price_data.index)

    def calculate_ma(self, level, window=30, matype=0):
        """
        Calculate Moving Average (MA) with configurable type.
        matype: 0=SMA, 1=EMA, 2=WMA, etc. (see TA-Lib docs)
        """
        price_data = self.get_level_data(level)
        return pd.Series(talib.MA(price_data['close'].values, timeperiod=window, matype=matype), index=price_data.index)
 
    def calculate_macd(self, level, fastperiod=12, slowperiod=26, signalperiod=9, 
                     fastmatype=MA_Type.EMA, slowmatype=MA_Type.EMA, signalmatype=MA_Type.EMA):
        """
        Calculate MACD with controllable MA types using MACDEXT
        By default uses EMA for all components
        Available MA types:
        - MA_Type.SMA : Simple Moving Average
        - MA_Type.EMA : Exponential Moving Averageapp/core/system/decision/processor.py
        
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
            price_data['close'].values,
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
       return pd.Series(talib.RSI(price_data['close'].values, timeperiod=window), index=price_data.index)

    def calculate_kama(self, level, window=30):
        price_data = self.get_level_data(level)
        return pd.Series(talib.KAMA(price_data['close'].values, timeperiod=window), index=price_data.index)
        
    def calculate_ht_trendline(self, level):
        """
        Hilbert Transform - Instantaneous Trendline
        Returns the trendline as a pandas Series
        """
        price_data = self.get_level_data(level)
        return pd.Series(talib.HT_TRENDLINE(price_data['close'].values), index=price_data.index)
        
    def calculate_ht_trendmode(self, level):
        """
        Hilbert Transform - Trend vs Cycle Mode
        Returns the integer values of the trend mode (1 for trend, 0 for cycle)
        """
        price_data = self.get_level_data(level)
        return pd.Series(talib.HT_TRENDMODE(price_data['close'].values), index=price_data.index)
        
    def calculate_natr(self, level, timeperiod=14):
        """
        Calculate Normalized Average True Range (NATR)
        
        Parameters:
        - level: The price level to use (e.g., 'close')
        - timeperiod: Number of periods to use for calculation (default: 14)
        
        Returns:
        - pandas.Series with NATR values
        """
        price_data = self.get_level_data(level)
        
        natr = talib.NATR(
            high=price_data['high'].values,
            low=price_data['low'].values,
            close=price_data['close'].values,
            timeperiod=timeperiod
        )
        return pd.Series(natr, index=price_data.index)
        
        
    def calculate_adosc_zscore(self, level, window=20, fastperiod=3, slowperiod=10):
        adosc = self.calculate_adosc(level, fastperiod, slowperiod)
        
        rolling_mean = adosc.rolling(window=window).mean()
        rolling_std = adosc.rolling(window=window).std()
        zscore = (adosc - rolling_mean) / rolling_std
        
        return zscore

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
        upper, middle, lower = talib.BBANDS(price_data['close'].values, timeperiod=window,
                                            nbdevup=nbdevup,
                                            nbdevdn=nbdevdn,
                                            matype=matype)
        return (
            pd.Series(upper, index=price_data.index),
            pd.Series(middle, index=price_data.index),
            pd.Series(lower, index=price_data.index)
        )

    def calculate_atr(self, level, window=14):
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.ATR(price_data['high'].values, price_data['low'].values, price_data['close'].values, timeperiod=window),
            index=price_data.index)

    def calculate_sar(self, level, acceleration=0.02, maximum=0.2):
        price_data = self.get_level_data(level)
        sar_values =  pd.Series(talib.SAR(price_data['high'].values, 
                                         price_data['low'].values,
                                         acceleration=acceleration, maximum=maximum
                                         ), index=price_data.index)
        
        sar_signal = pd.Series(index=price_data.index, dtype='int')
        sar_signal[price_data['close'] > sar_values] = 1
        sar_signal[price_data['close'] < sar_values] = -1
        return sar_values, sar_signal

    def calculate_stochrsi(self, level, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        price_data = self.get_level_data(level)
        fastk, fastd = talib.STOCHRSI(
            price_data['close'].values,
            timeperiod=timeperiod,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
            fastd_matype=fastd_matype
        )
        return (
            pd.Series(fastk, index=price_data.index),
            pd.Series(fastd, index=price_data.index)
        )
        
    def calculate_minmax(self, level, window = 12):
        price_data = self.get_level_data(level)
        min, max = talib.MINMAX(price_data['close'].values, timeperiod=window)
        return (
            pd.Series(min, index=price_data.index),
            pd.Series(max, index=price_data.index)
        )
        
    def calculate_ppo(self, level, fastperiod=12, slowperiod=26, matype=0):
        price_data = self.get_level_data(level)
        return pd.Series(talib.PPO(price_data['close'].values,
                                   fastperiod=fastperiod,
                                   slowperiod=slowperiod,
                                   matype=matype),
                         index=price_data.index)

    def calculate_donchian(self, level, window=20):
        """
        Calculate Donchian Channels
        Returns upper, middle and lower bands
        Upper band = highest high over N periods
        Lower band = lowest low over N periods
        Middle band = (upper + lower) / 2
        """
        price_data = self.get_level_data(level)
        
        # Calculate rolling highest high and lowest low
        upper = price_data['high'].rolling(window=window).max()
        lower = price_data['low'].rolling(window=window).min()
        middle = (upper + lower) / 2
        
        return (
            pd.Series(upper, index=price_data.index),
            pd.Series(middle, index=price_data.index),
            pd.Series(lower, index=price_data.index)
        )
        
    def calculate_sar_ext(self, level,
                        startvalue=0.0, offsetonreverse=0.0,
                        accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
                        accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2):
        """
        Calculate Parabolic SAR Extended (SAREXT)
        """
        price_data = self.get_level_data(level)
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
    
    
    def apply_macd_mama(self, name, level, fastlimit=0.5, slowlimit=0.05, signalperiod=9):
        mama, fama = self.calculate_mama(level=level, fastlimit=fastlimit, slowlimit=slowlimit)
        line = mama - fama
        signal = line.rolling(signalperiod).mean()
        histogram = line - signal
        
        self.data[f'{name}_line_{level}'] = self.data.index.get_level_values(level).map(line)
        self.data[f'{name}_signal_{level}'] = self.data.index.get_level_values(level).map(signal)
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
        
        self.data[f'{name}_upper_close_{level}'] = 100*(self.data[f'{name}_upper_{level}'] - self.data['close']) / self.data['close']
        self.data[f'{name}_lower_close_{level}'] = 100*(self.data[f'{name}_lower_{level}'] - self.data['close']) / self.data['close']
    
    def apply_atr(self, name, level, window=14):
        atr = self.calculate_atr(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(atr)
    
    def apply_sar(self, name, level, acceleration=0.02, maximum=0.2):
        sar_values, sar_signal = self.calculate_sar(level, acceleration, maximum)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(sar_values)
        self.data[f'{name}_signal_{level}'] = self.data.index.get_level_values(level).map(sar_signal)
        
    
    def apply_stochrsi(self, name, level, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        fastk, fastd = self.calculate_stochrsi(level, timeperiod, fastk_period, fastd_period, fastd_matype)
        self.data[f'{name}_k_{level}'] = self.data.index.get_level_values(level).map(fastk)
        self.data[f'{name}_d_{level}'] = self.data.index.get_level_values(level).map(fastd)

    
    def calculate_stoch(self, level, fastk_period=5, slowk_period=3, slowd_period=3, slowk_matype=0, slowd_matype=0):
        price_data = self.get_level_data(level)
        slowk, slowd = talib.STOCH(price_data['high'].values, price_data['low'].values, price_data['close'].values,
                                   fastk_period=fastk_period,
                                   slowk_period=slowk_period, slowk_matype=slowk_matype,
                                   slowd_period=slowd_period, slowd_matype=slowd_matype)
        return (
            pd.Series(slowk, index=price_data.index),
            pd.Series(slowd, index=price_data.index)
        )
    
    def apply_stoch(self, name, level, fastk_period=5, slowk_period=3, slowd_period=3, slowk_matype=0, slowd_matype=0):
        slowk, slowd = self.calculate_stoch(level, fastk_period, slowk_period, slowd_period, slowk_matype, slowd_matype)
        self.data[f'{name}_slowk_{level}'] = self.data.index.get_level_values(level).map(slowk)
        self.data[f'{name}_slowd_{level}'] = self.data.index.get_level_values(level).map(slowd)

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
        
    def calculate_ad(self, level):
        """Calculate Chaikin A/D Line"""
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.AD(price_data['high'], price_data['low'], price_data['close'], price_data['volume']),
            index=price_data.index
        )

    def apply_ad(self, name, level):
        """Apply Chaikin A/D Line to dataframe"""
        ad = self.calculate_ad(level)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(ad)

    def calculate_adosc(self, level, fastperiod=3, slowperiod=10):
        """Calculate Chaikin Oscillator"""
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.ADOSC(price_data['high'], price_data['low'], price_data['close'], price_data['volume'], 
                       fastperiod=fastperiod, slowperiod=slowperiod),
            index=price_data.index
        )

    def apply_adosc(self, name, level, fastperiod=3, slowperiod=10):
        """Apply Chaikin Oscillator to dataframe"""
        adosc = self.calculate_adosc(level, fastperiod, slowperiod)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(adosc)

    def calculate_obv(self, level):
        """Calculate On Balance Volume"""
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.OBV(price_data['close'], price_data['volume']),
            index=price_data.index
        )

    def apply_obv(self, name, level):
        """Apply On Balance Volume to dataframe"""
        obv = self.calculate_obv(level)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(obv)
        
    def apply_minmax(self, name, level, window):
        min, max = self.calculate_minmax(level, window)
        self.data[f'{name}_min_{level}'] = self.data.index.get_level_values(level).map(min)
        self.data[f'{name}_max_{level}'] = self.data.index.get_level_values(level).map(max)

    def apply_ht_trendline(self, name, level):
        trendline = self.calculate_ht_trendline(level)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(trendline)
        return self.data
        
    def apply_ht_trendmode(self, name, level):
        trendmode = self.calculate_ht_trendmode(level)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(trendmode)
        return self.data
        
        
    def apply_natr(self, name, level, timeperiod=14):
        natr = self.calculate_natr(level, timeperiod)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(natr)
        return self.data
        
    def apply_adosc_zscore(self, name, level, window=20, fastperiod=3, slowperiod=10):
        adosc_zscore = self.calculate_adosc_zscore(level, window, fastperiod, slowperiod)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(adosc_zscore)
        return self.data
        
    def calculate_sar_distance(self, level, acceleration, maximum):
        sar_values, sar_signal = self.calculate_sar(level, acceleration, maximum)
        price_data = self.get_level_data(level)
        price = (price_data['close'] + price_data['open']) / 2
        distance = ((price - sar_values) / price) * 100
        return pd.Series(distance, index=price_data.index)
            
    def apply_sar_distance(self, name, level, acceleration, maximum):
        distance = self.calculate_sar_distance(level, acceleration, maximum)
        self.data[f'{name}_distance_{level}'] = self.data.index.get_level_values(level).map(distance)
        return self.data
        
    
    def calculate_sar_zscore(self, level, acceleration, maximum, window=20):
        distance = self.calculate_sar_distance(level, acceleration, maximum)
        rolling_mean = distance.rolling(window=window).mean()
        rolling_std = distance.rolling(window=window).std()
        zscore = (distance - rolling_mean) / rolling_std
        return zscore
    
    
    def apply_sar_zscore(self, name, level, acceleration, maximum, window=20):
        zscore = self.calculate_sar_zscore(level, acceleration, maximum, window)
        self.data[f'{name}_zscore_{level}'] = self.data.index.get_level_values(level).map(zscore)
        return self.data
    
    
    
    
    
    
    
    
    
    def calculate_htrendline_close_distance(self, level_ht, level_close):
        trendline = self.calculate_ht_trendline(level_ht)
        trendline = self.data.index.get_level_values(level_ht).map(trendline)
        price_data = self.get_level_data(level_close)
        distance = ((price_data['close'] - trendline) / price_data['close']) * 100
        return pd.Series(distance, index = price_data['close'].index)
    
    def apply_htrendline_close_distance(self, name, level_ht, level_close):
        distance = self.calculate_htrendline_close_distance(level_ht, level_close)
        self.data[f'{name}_distance_{level_ht}_{level_close}'] = self.data.index.get_level_values(level_close).map(distance)
        return self.data
    
    
    def calculate_htrendline_close_zscore(self, level_ht, level_close, window):
        distance = self.calculate_htrendline_close_distance(level_ht, level_close)
        rolling_mean = distance.rolling(window=window).mean()
        rolling_std = distance.rolling(window=window).std()
        zscore = (distance - rolling_mean) / rolling_std
        return pd.Series(zscore, index = distance.index)
    
    def apply_htrendline_close_zscore(self, name, level_ht, level_close, window):
        zscore = self.calculate_htrendline_close_zscore(level_ht, level_close, window)
        self.data[f'{name}_zscore_{level_ht}_{level_close}'] = self.data.index.get_level_values(level_close).map(zscore)
        return self.data
    

    # Momementum
    def calculate_adx(self, level, window=14):
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.ADX(
                price_data['high'].values,
                price_data['low'].values,
                price_data['close'].values,
                timeperiod=window
            ),
            index=price_data.index
        )
    
    def apply_adx(self, name, level, window=14):
        adx = self.calculate_adx(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(adx)

    # Momentum Indicators #####################################################
    
    def calculate_cmo(self, level, window=14):
        """Calculate Chande Momentum Oscillator (CMO)"""
        price_data = self.get_level_data(level)
        return pd.Series(talib.CMO(price_data['close'].values, timeperiod=window), index=price_data.index)

    def apply_cmo(self, name, level, window=14):
        cmo = self.calculate_cmo(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(cmo)

    def calculate_cci(self, level, window=14):
        """Calculate Commodity Channel Index (CCI)"""
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.CCI(price_data['high'], price_data['low'], price_data['close'], timeperiod=window),
            index=price_data.index
        )

    def apply_cci(self, name, level, window=14):
        cci = self.calculate_cci(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(cci)

    def calculate_dx(self, level, window=14):
        """Calculate Directional Movement Index (DX)"""
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.DX(price_data['high'], price_data['low'], price_data['close'], timeperiod=window),
            index=price_data.index
        )

    def apply_dx(self, name, level, window=14):
        dx = self.calculate_dx(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(dx)

    def calculate_ultosc(self, level, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        """Calculate Ultimate Oscillator (ULTOSC)"""
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.ULTOSC(price_data['high'], price_data['low'], price_data['close'],
                        timeperiod1=timeperiod1,
                        timeperiod2=timeperiod2,
                        timeperiod3=timeperiod3),
            index=price_data.index
        )

    def apply_ultosc(self, name, level, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        ultosc = self.calculate_ultosc(level, timeperiod1, timeperiod2, timeperiod3)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(ultosc)

    def calculate_rocp(self, level, window=10):
        """Calculate Momentum Indicator (MOM)"""
        price_data = self.get_level_data(level)
        return pd.Series(100*talib.ROCP(price_data['close'].values, timeperiod=window), index=price_data.index)

    def apply_rocp(self, name, level, window=10):
        rocp = self.calculate_rocp(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(rocp)

    def calculate_mom(self, level, period):
        price_data = self.get_level_data(level)
        return pd.Series(price_data['close'].pct_change().rolling(period).mean(), index=price_data.index)

    def calculate_mom_zscore(self, level, period, window=20):
        mom = self.calculate_mom(level, period) * 100
        rolling_mean = mom.rolling(window=window).mean()
        rolling_std = mom.rolling(window=window).std()
        zscore = (mom - rolling_mean) / rolling_std
        return zscore

    def apply_mom_zscore(self, name, level, period, window=20):
        zscore = self.calculate_mom_zscore(level, period, window)
        self.data[f'{name}_zscore_{level}'] = self.data.index.get_level_values(level).map(zscore)

    def calculate_rocp_zscore(self, level, window=10, zscore_window=20):
        rocp = self.calculate_rocp(level, window)
        rolling_mean = rocp.rolling(window=zscore_window).mean()
        rolling_std = rocp.rolling(window=zscore_window).std()
        zscore = (rocp - rolling_mean) / rolling_std
        return zscore

    def apply_rocp_zscore(self, name, level, window=10, zscore_window=20):
        zscore = self.calculate_rocp_zscore(level, window, zscore_window)
        self.data[f'{name}_zscore_{level}'] = self.data.index.get_level_values(level).map(zscore)

    def apply_mom(self, name, level, period, period_sign):
        mom = self.calculate_mom(level, period) * 100
        mom_sign = mom.rolling(period_sign).mean()
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(mom)
        self.data[f'{name}_sign_{level}'] = self.data.index.get_level_values(level).map(mom_sign)


    def calculate_plus_di(self, level, window=14):
        """Calculate Plus Directional Indicator (PLUS_DI)"""
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.PLUS_DI(price_data['high'], price_data['low'], price_data['close'], timeperiod=window),
            index=price_data.index
        )
    
    def calculate_plus_dm(self, level, window=14):
        """Calculate Plus Directional Movement (PLUS_DM)"""
        price_data = self.get_level_data(level)
        return pd.Series(
            talib.PLUS_DM(price_data['high'], price_data['low'], timeperiod=window),
            index=price_data.index
        )

    def apply_plus_dm(self, name, level, window=14):
        plus_dm = self.calculate_plus_dm(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(plus_dm)

    def apply_plus_di(self, name, level, window=14):
        plus_di = self.calculate_plus_di(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(plus_di)

    # Linear Regression Indicators ############################################
    def calculate_linearreg(self, level, window=14):
        """Calculate Linear Regression"""
        price_data = self.get_level_data(level)
        return pd.Series(talib.LINEARREG(price_data['close'].values, timeperiod=window), index=price_data.index)

    def apply_linearreg(self, name, level, window=14):
        linearreg = self.calculate_linearreg(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(linearreg)

    def calculate_linearreg_angle(self, level, window=14):
        """Calculate Linear Regression Angle"""
        price_data = self.get_level_data(level)
        return pd.Series(talib.LINEARREG_ANGLE(price_data['close'].values, timeperiod=window), index=price_data.index)

    def apply_linearreg_angle(self, name, level, window=14):
        angle = self.calculate_linearreg_angle(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(angle)

    def calculate_linearreg_intercept(self, level, window=14):
        """Calculate Linear Regression Intercept"""
        price_data = self.get_level_data(level)
        return pd.Series(talib.LINEARREG_INTERCEPT(price_data['close'].values, timeperiod=window), index=price_data.index)

    def apply_linearreg_intercept(self, name, level, window=14):
        intercept = self.calculate_linearreg_intercept(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(intercept)

    def calculate_linearreg_slope(self, level, window=14):
        """Calculate Linear Regression Slope"""
        price_data = self.get_level_data(level)
        return pd.Series(talib.LINEARREG_SLOPE(price_data['close'].values, timeperiod=window), index=price_data.index)

    def apply_linearreg_slope(self, name, level, window=14):
        slope = self.calculate_linearreg_slope(level, window)
        self.data[f'{name}_{level}'] = self.data.index.get_level_values(level).map(slope)
