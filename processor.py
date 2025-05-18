from tools.indicator import Indicator, MA_Type


class AddIndicator:
    
    def __init__(self, data):
        self.data = data
        self.indicator = Indicator(data)
        self.timeframes = {'primary': '5m', 'second': '15m',
                           'trend': '30m', 'long': '1h'}
        self.params = {
            'fast': 12, 'ratio_fast_slow':6,
            'signal':9, 'atr': 14,
            'sar_acceleration': 0.02,
            'sar_max_acceleration': 0.2,
            'mama_fastlimit': 0.5,
            'mama_slowlimit': 0.05,
            'ultosc' : (12*2, 36*2, 72*2),
            'score_window_5m': 12*24*5
        }
        
    def update_params(self, params):
        self.params = params
        self.params['slow'] = self.params['fast'] * self.params['ratio_fast_slow']
        self.params['bb_period'] = self.params['slow']
        self.params['bb_std'] = 2
    
    def base_indicator(self):
        # Add sessions
        self.indicator.add_session()
        
        # MAMA
        self.indicator.apply_mama('mama', self.timeframes['primary'],
                                 fastlimit=self.params['mama_fastlimit'],
                                 slowlimit=self.params['mama_slowlimit']
                                 )
        self.indicator.apply_mama('mama', self.timeframes['second'],
                                 fastlimit=self.params['mama_fastlimit'],
                                 slowlimit=self.params['mama_slowlimit']
                                 )
        self.indicator.apply_mama('mama', self.timeframes['trend'],
                                 fastlimit=self.params['mama_fastlimit'],
                                 slowlimit=self.params['mama_slowlimit']
                                 )
        self.indicator.apply_mama('mama', self.timeframes['long'],
                                 fastlimit=self.params['mama_fastlimit'],
                                 slowlimit=self.params['mama_slowlimit']
                                 )
        self.indicator.apply_mama('mama', '12h',
                                 fastlimit=self.params['mama_fastlimit'],
                                 slowlimit=self.params['mama_slowlimit']
                                 )
        
        # MACD        
        self.indicator.apply_macd_mama(name='macd_mama', level=self.timeframes['primary'], 
                               fastlimit=self.params['mama_fastlimit'], 
                               slowlimit=self.params['mama_slowlimit'], 
                               signalperiod=self.params['signal']
                               )
        self.indicator.apply_macd_mama(name='macd_mama', level=self.timeframes['second'], 
                               fastlimit=self.params['mama_fastlimit'], 
                               slowlimit=self.params['mama_slowlimit'], 
                               signalperiod=self.params['signal']
                               )
        self.indicator.apply_macd_mama(name='macd_mama', level=self.timeframes['trend'], 
                               fastlimit=self.params['mama_fastlimit'], 
                               slowlimit=self.params['mama_slowlimit'], 
                               signalperiod=self.params['signal']
                               )
        self.indicator.apply_macd_mama(name='macd_mama', level=self.timeframes['long'], 
                               fastlimit=self.params['mama_fastlimit'], 
                               slowlimit=self.params['mama_slowlimit'], 
                               signalperiod=self.params['signal']
                               )
        self.indicator.apply_macd_mama(name='macd_mama', level='12h', 
                               fastlimit=self.params['mama_fastlimit'], 
                               slowlimit=self.params['mama_slowlimit'], 
                               signalperiod=self.params['signal']
                               )
        
        # SAR
        self.indicator.apply_sar('sar', self.timeframes['primary'], 
                                  self.params['sar_acceleration'],
                                  self.params['sar_max_acceleration']
                                  )
        self.indicator.apply_sar('sar', self.timeframes['second'], 
                                  self.params['sar_acceleration'],
                                  self.params['sar_max_acceleration']
                                  )
        self.indicator.apply_sar('sar', self.timeframes['trend'], 
                                  self.params['sar_acceleration'],
                                  self.params['sar_max_acceleration']
                                  )
        self.indicator.apply_sar('sar', self.timeframes['long'], 
                                  self.params['sar_acceleration'],
                                  self.params['sar_max_acceleration']
                                  )
        self.indicator.apply_sar('sar', '4h', 
                                  self.params['sar_acceleration'],
                                  self.params['sar_max_acceleration']
                                  )
        self.indicator.apply_sar('sar', '8h', 
                                  self.params['sar_acceleration'],
                                  self.params['sar_max_acceleration']
                                  )
        self.indicator.apply_sar('sar', '12h', 
                                  self.params['sar_acceleration'],
                                  self.params['sar_max_acceleration']
                                  )
        
        # DC
        self.indicator.apply_donchian('donchian', self.timeframes['long'], 8)
        
        # ht_trendline
        self.indicator.apply_ht_trendline('ht_trendline', self.timeframes['primary'])
        self.indicator.apply_ht_trendline('ht_trendline', self.timeframes['second'])
        self.indicator.apply_ht_trendline('ht_trendline', self.timeframes['trend'])
        self.indicator.apply_ht_trendline('ht_trendline', self.timeframes['long'])
        self.indicator.apply_ht_trendline('ht_trendline', '4h')
        self.indicator.apply_ht_trendline('ht_trendline', '8h')
        self.indicator.apply_ht_trendline('ht_trendline', '12h')
        self.indicator.apply_ht_trendline('ht_trendline', '1d')
        
        
        # Score / distance
        self.indicator.apply_sar_distance('sar', self.timeframes['primary'],
                                                self.params['sar_acceleration'], self.params['sar_max_acceleration'])
        self.indicator.apply_sar_distance('sar', self.timeframes['second'],
                                                self.params['sar_acceleration'], self.params['sar_max_acceleration'])
        self.indicator.apply_sar_distance('sar', self.timeframes['trend'],
                                                self.params['sar_acceleration'], self.params['sar_max_acceleration'])
        self.indicator.apply_sar_distance('sar', self.timeframes['long'],
                                                self.params['sar_acceleration'], self.params['sar_max_acceleration'])
        self.indicator.apply_sar_distance('sar', '4h',
                                                self.params['sar_acceleration'], self.params['sar_max_acceleration'])
        self.indicator.apply_sar_distance('sar', '8h',
                                                self.params['sar_acceleration'], self.params['sar_max_acceleration'])
        self.indicator.apply_sar_distance('sar', '12h',
                                                self.params['sar_acceleration'], self.params['sar_max_acceleration'])
        
        
        
        
        self.indicator.apply_sar_zscore('sar', self.timeframes['primary'],
                                              acceleration=self.params['sar_acceleration'],
                                              maximum=self.params['sar_max_acceleration'],
                                              window=self.params['score_window_5m'])
        
        
        self.indicator.apply_htrendline_close_distance('ht_trendline_close',
                                                       level_ht=self.timeframes['trend'],
                                                       level_close=self.timeframes['primary'])
        self.indicator.apply_htrendline_close_zscore('ht_trendline_close',
                                                     level_ht=self.timeframes['trend'],
                                                     level_close=self.timeframes['primary'],
                                                     window=self.params['score_window_5m'])
        
        self.indicator.apply_htrendline_close_distance('ht_trendline_close',
                                                       level_ht=self.timeframes['long'],
                                                       level_close=self.timeframes['primary'])
        self.indicator.apply_htrendline_close_zscore('ht_trendline_close',
                                                     level_ht=self.timeframes['long'],
                                                     level_close=self.timeframes['primary'],
                                                     window=self.params['score_window_5m'])
        
        self.indicator.apply_htrendline_close_distance('ht_trendline_close',
                                                       level_ht='12h',
                                                       level_close=self.timeframes['primary'])
        self.indicator.apply_htrendline_close_zscore('ht_trendline_close',
                                                     level_ht='12h',
                                                     level_close=self.timeframes['primary'],
                                                     window=self.params['score_window_5m'])
    
    def supp_indicator(self):
        self.indicator.apply_minmax(name='MM', level=self.timeframes['long'], window=8)
        
        # Bollinger Bands
        self.indicator.apply_bollinger('bb', self.timeframes['primary'], 
                                    self.params['bb_period'], 
                                    self.params['bb_std'],
                                    matype=MA_Type.WMA)
        self.indicator.apply_bollinger('bb', self.timeframes['trend'], 
                                    self.params['bb_period'], 
                                    self.params['bb_std'],
                                    matype=MA_Type.WMA)
        self.indicator.apply_bollinger('bb', self.timeframes['long'], 
                                    self.params['bb_period'], 
                                    self.params['bb_std'],
                                    matype=MA_Type.WMA)
        
        # Volatity
        self.indicator.apply_atr(name='atr', level=self.timeframes['long'],
                                 window=self.params['atr'])
        self.indicator.apply_natr('natr', self.timeframes['trend'], self.params['atr'])
        
        # Volume
        self.indicator.apply_obv(name='obv', level=self.timeframes['primary'])
        self.indicator.apply_adosc('adosc', level=self.timeframes['primary'],
                                   fastperiod=self.params['fast'],
                                   slowperiod=self.params['slow'])
        
        # Mom
        
        self.indicator.apply_stoch('stoch', self.timeframes['primary'],
                                   fastk_period=self.params['slow'],
                                   slowk_period=self.params['fast'],
                                   slowd_period=self.params['ratio_fast_slow'],
                                   slowk_matype=0, slowd_matype=0)
        
        self.indicator.apply_stochrsi('stochrsi', self.timeframes['primary'],
                                      timeperiod=self.params['atr'],
                                      fastk_period=self.params['slow'],
                                      fastd_period=self.params['ratio_fast_slow'],
                                      fastd_matype=0)
        
        self.indicator.apply_mom('mom', self.timeframes['primary'], period=self.params['atr'], period_sign=9)
        self.indicator.apply_rocp('rocp', self.timeframes['primary'], self.params['atr'])
        self.indicator.apply_dx('dx', self.timeframes['primary'], self.params['atr'])
        self.indicator.apply_rsi('rsi', self.timeframes['trend'], self.params['atr'])
        self.indicator.apply_ultosc('ultosc', self.timeframes['trend'], 
                                    timeperiod1=self.params['ultosc'][0], 
                                    timeperiod2=self.params['ultosc'][1],
                                    timeperiod3=self.params['ultosc'][2]
                                    )
        
        # Add z-score indicators
        self.indicator.apply_mom_zscore('mom', self.timeframes['primary'], period=self.params['atr'], window=self.params['score_window_5m'])
        self.indicator.apply_rocp_zscore('rocp', self.timeframes['primary'], window=self.params['atr'], zscore_window=self.params['score_window_5m'])
        
        self.indicator.apply_adosc_zscore('adosc_zscore', level=self.timeframes['primary'],
                                          fastperiod=self.params['fast'],
                                          slowperiod=self.params['slow'],
                                          window=self.params['score_window_5m'])
    
        
    def apply(self):
        self.base_indicator()
        self.supp_indicator()
    
    