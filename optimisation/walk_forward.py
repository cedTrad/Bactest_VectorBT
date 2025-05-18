import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp

from .base import OptimizerSimple
from sampling import Sampling

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
        self.results = []

        for i, ((train_start, train_end), (test_start, test_end)) in enumerate(splits):
            train_data = self.data[(self.data.index.get_level_values('5m') >= train_start) & (self.data.index.get_level_values('5m') <= train_end)]
            test_data = self.data[(self.data.index.get_level_values('5m') >= test_start) & (self.data.index.get_level_values('5m') <= test_end)]
            
            # Optimisation sur train avec validation croisée interne
            optimizer = OptimizerSimple(self.strategy_class, train_data)
            train_rslt = optimizer.cross_val_optimize(n_trials=n_trials, n_splits=3)
            best_params = train_rslt['best_params']

            # Test sur test
            strategy = self.strategy_class(test_data.copy())
            pf = strategy.run(best_params['fast'], best_params['slow'])
            test_sharpe = pf.sharpe_ratio()
            train_profit = pf.total_return()

            self.results.append({
                'split': i,
                'train_period_0': train_start,
                'train_period_1': train_end,
                'test_period_0': test_start,
                'test_period_1':  test_end,
                'train_metrics': {
                    'cv_scores': train_rslt['cv_scores'],
                    'avg_score': train_rslt['avg_score']
                },
                'test_metrics': {'sharpe': test_sharpe, 'profit': train_profit},
                'best_params': best_params
            })
            print(f"Split {i}: Sharpe={test_sharpe:.2f}, Params={best_params}")

        return self.results
    
    def analyze_rslt(self):
        results = self.results

        profit_train = [rslt['train_metrics']['avg_score'] for rslt in results]
        sharpe_train = [rslt['train_metrics']['avg_score'] for rslt in results]
        profit_test = [rslt['test_metrics']['profit'] for rslt in results]
        sharpe_test = [rslt['test_metrics']['sharpe'] for rslt in results]
        fast = [rslt['best_params']['fast'] for rslt in results]
        slow = [rslt['best_params']['slow'] for rslt in results]

        splits = list(range(len(results)))
        df = pd.DataFrame({
            'split': splits,
            'profit_train': profit_train,
            'profit_test': profit_test,
            'sharpe_train': sharpe_train,
            'sharpe_test': sharpe_test,
            'fast': fast,
            'slow': slow
        })

        # Analyse de la stabilité des paramètres
        import numpy as np
        fast_std = np.std(fast)
        slow_std = np.std(slow)
        print(f"Stabilité des paramètres : fast std={fast_std:.2f}, slow std={slow_std:.2f}")
        
        # Visualisation automatique avec Plotly
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Evolution des paramètres optimaux", "Performance Out-of-Sample"))
        # Paramètres
        fig.add_trace(go.Scatter(x=splits, y=fast, mode='lines+markers', name='fast'), row=1, col=1)
        fig.add_trace(go.Scatter(x=splits, y=slow, mode='lines+markers', name='slow'), row=1, col=1)
        # Performances
        fig.add_trace(go.Scatter(x=splits, y=sharpe_test, mode='lines+markers', name='Sharpe Test', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=splits, y=profit_test, mode='lines+markers', name='Profit Test', line=dict(color='orange')), row=2, col=1)
        fig.update_xaxes(title_text="Split", row=2, col=1)
        fig.update_yaxes(title_text="Paramètres", row=1, col=1)
        fig.update_yaxes(title_text="Performance Test", row=2, col=1)
        fig.update_layout(height=700, width=900, title_text="Analyse Walk-Forward (paramètres & performance)")
        fig.show()

        return df