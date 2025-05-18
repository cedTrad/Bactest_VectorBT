import vectorbt as vbt
import plotly.graph_objects as go

    

class SarStrategy:
    def __init__(self, data, capital=100, leverage=10):
        self.data = data.copy()
        self.capital = capital
        self.leverage = leverage
        self.price = self.data['close_30min']  # par exemple, price aligné sur '30m'
        self.pf = None

    def generate_signals(self):
        trend_1 = (
            (self.data['sar_signal_12h'] == 1) &
            (self.data['sar_signal_8h'] == 1) &
            (self.data['sar_signal_4h'] == 1)
        )
        entry_signal = trend_1 & (self.data['sar_signal_30m'] == 1)
        exit_signal = ~entry_signal
        
        self.entries = entry_signal.astype(bool)
        self.exits = exit_signal.astype(bool)

    def run(self):
        self.generate_signals()
        
        self.pf = vbt.Portfolio.from_signals(close=self.price, entries=self.entries, exits=self.exits,
                                             init_cash=self.capital
        )
        return self.pf

    def viz(self):
        fig = self.price.vbt.plot(trace_kwargs=dict(name='Price'), height=500, width=1000)
        self.pf.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)
        fig.show()

        self.pf.plot().show()
