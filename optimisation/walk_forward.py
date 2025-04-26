
from .base import OptimizerSimple
from utils import OptimConfig



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