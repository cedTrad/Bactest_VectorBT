import vectorbt as vbt
from processor import Processor
import plotly.graph_objects as go


# 1. Classe pour la stratégie
class KamaStrategy:
    def __init__(self, data):
        self.data = data.copy()
        self.processor = None

    def apply_kama(self, fast, slow):
        self.processor = Processor(self.data)
        self.processor.apply_kama(name='kama_fast', level='5m', window=fast)
        self.processor.apply_kama(name='kama_slow', level='5m', window=slow)
        self.data = self.data.dropna()

    def run(self, fast, slow):
        self.apply_kama(fast, slow)
        price = self.data.groupby(level='5m')['close'].last()
        kama_fast = self.data.groupby(level='5m')['kama_fast_5m'].last()
        kama_slow = self.data.groupby(level='5m')['kama_slow_5m'].last()
        entries = kama_fast > kama_slow
        exits = kama_fast < kama_slow
        self.pf = vbt.Portfolio.from_signals(price, entries, exits)
        return self.pf
    
    def viz(self, fast, slow):
        # Appliquer la stratégie et obtenir le portefeuille
        fast_ma = self.data.groupby(level='5m')['kama_fast_5m'].last()
        slow_ma = self.data.groupby(level='5m')['kama_slow_5m'].last()

        fig = self.pf.positions.plot()
        fig.show()
        
    
