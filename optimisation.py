import optuna
import vectorbt as vbt

from processor import Processor
from sampling import Sampling, OptimConfig



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
        pf = vbt.Portfolio.from_signals(price, entries, exits)
        return pf
    
    def viz(self, fast, slow):
        # Appliquer la stratégie et obtenir le portefeuille
        pf = self.run(fast, slow)
        price = self.data.groupby(level='5m')['close'].last()
        fast_ma = self.data.groupby(level='5m')['kama_fast_5m'].last()
        slow_ma = self.data.groupby(level='5m')['kama_slow_5m'].last()
        
        fig = price.vbt.plot(trace_kwargs=dict(name='Close'))
        fast_ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=fig)
        slow_ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=fig)
        pf.plot(fig=fig)
        

# 2. Classe pour l'optimisation simple
class OptimizerSimple:
    def __init__(self, strategy_class, data):
        self.strategy_class = strategy_class
        self.data = data.copy()

    def objective(self, trial):
        fast = trial.suggest_int('fast', 2, 20)
        slow = trial.suggest_int('slow', 20, 50)
        strategy = self.strategy_class(self.data.copy())
        pf = strategy.run(fast, slow)
        return pf.sharpe_ratio()

    def optimize(self, n_trials):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print("Best params:", study.best_params)
        return study.best_params

# 3. Classe pour l'optimisation walk-forward
class WalkForwardOptimizer:
    def __init__(self, start, end, optim_config, strategy_class, data):
        self.start = start
        self.end = end
        self.optim_config = optim_config
        self.sampler = Sampling(start, end, optim_config)
        self.strategy_class = strategy_class
        self.data = data.copy()

    def walk_forward(self, n_trials=20, verbose=True):
        
        splits = self.sampler.generate_splits()
        results = []

        for i, ((train_start, train_end), (test_start, test_end)) in enumerate(splits):
            train_data = self.data[(self.data.index.get_level_values('5m') >= train_start) & (self.data.index.get_level_values('5m') <= train_end)]
            test_data = self.data[(self.data.index.get_level_values('5m') >= test_start) & (self.data.index.get_level_values('5m') <= test_end)]
            
            # Optimisation sur train
            optimizer = OptimizerSimple(self.strategy_class, train_data)
            best_params = optimizer.optimize(n_trials=n_trials)

            # Test sur test
            strategy = self.strategy_class(test_data.copy())
            pf = strategy.run(best_params['fast'], best_params['slow'])
            sharpe = pf.sharpe_ratio()

            results.append({
                'split': i,
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end),
                'best_params': best_params,
                'test_sharpe': sharpe
            })
            print(f"Split {i}: Sharpe={sharpe:.2f}, Params={best_params}")

        return results