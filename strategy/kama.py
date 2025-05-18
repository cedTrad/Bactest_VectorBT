import vectorbt as vbt
from processor import Processor
import plotly.graph_objects as go


# 1. Classe pour la stratégie
class KamaStrategy:
    def __init__(self, data, capital = 100):
        self.capital = capital
        self.data = data.copy()
        self.processor = None

    def apply_kama(self, fast, slow):
        self.processor = Processor(self.data)
        self.processor.apply_kama(name='kama_fast', level='5m', window=fast)
        self.processor.apply_kama(name='kama_slow', level='5m', window=slow)
        self.data = self.data.dropna()

    def run(self, fast, slow):
        self.apply_kama(fast, slow)
        
        self.price = self.data.groupby(level='5m')['close'].last()
        self.kama_fast = self.data.groupby(level='5m')['kama_fast_5m'].last()
        self.kama_slow = self.data.groupby(level='5m')['kama_slow_5m'].last()
        
        entries = self.kama_fast > self.kama_slow
        exits = self.kama_fast < self.kama_slow
        self.pf = vbt.Portfolio.from_signals(self.price, entries, exits, init_cash=self.capital)
        return self.pf
    
    def viz(self):
        fig = self.price.vbt.plot(trace_kwargs=dict(name='Close'), height=500, width=1000)
        self.kama_fast.vbt.plot(trace_kwargs=dict(name='Fast KAMA'), fig=fig)
        self.kama_slow.vbt.plot(trace_kwargs=dict(name='Slow KAMA'), fig=fig)
        self.pf.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)
        fig.show()
        
        self.pf.plot().show()
    

class SAR:
    def __init__(self, data, capital = 100):
        self.capital = capital
        self.data = data.copy()
        self.processor = None

    def apply_kama(self, fast, slow):
        self.processor = Processor(self.data)
        self.processor.apply_kama(name='kama_fast', level='5m', window=fast)
        self.processor.apply_kama(name='kama_slow', level='5m', window=slow)
        self.data = self.data.dropna()

    def run(self, fast, slow):
        self.apply_kama(fast, slow)
        
        self.price = self.data.groupby(level='5m')['close'].last()
        self.kama_fast = self.data.groupby(level='5m')['kama_fast_5m'].last()
        self.kama_slow = self.data.groupby(level='5m')['kama_slow_5m'].last()
        
        entries = self.kama_fast > self.kama_slow
        exits = self.kama_fast < self.kama_slow
        self.pf = vbt.Portfolio.from_signals(self.price, entries, exits, init_cash=self.capital)
        return self.pf
    