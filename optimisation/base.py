import numpy as np
import optuna


# 2. Classe pour l'optimisation simple
class OptimizerSimple:
    def __init__(self, strategy_class, data):
        self.strategy_class = strategy_class
        self.data = data.copy()
        self.metrics = {}
        self.result = {}

    def objective(self, trial):
        fast = trial.suggest_int('fast', 2, 12*4)
        slow = trial.suggest_int('slow', 12*4, 12*12)
        strategy = self.strategy_class(self.data.copy())
        pf = strategy.run(fast, slow)
        
        self.metrics['profit'] = pf.total_return()
        self.metrics['sharpe'] = pf.sharpe_ratio()
        return self.metrics['profit']

    def optimize(self, n_trials):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print("Best params:", study.best_params)
        
        self.result = {'study' : study,
                  'best_params' : study.best_params,
                  'metrics': self.metrics}
        
        return self.result
    
    def cross_val_optimize(self, n_trials, n_splits=3):
        """
        Validation croisée interne sur le train (time series split)
        """
        data = self.data
        split_size = len(data) // n_splits
        scores = []
        params = []

        for i in range(n_splits):
            train_end = (i + 1) * split_size
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:train_end + split_size]

            # Optimisation sur train_data
            optimizer = OptimizerSimple(self.strategy_class, train_data)
            rslt = optimizer.optimize(n_trials=n_trials)
            best_params = rslt['best_params']

            # Test sur val_data
            strategy = self.strategy_class(val_data.copy())
            pf = strategy.run(best_params['fast'], best_params['slow'])
            val_score = pf.total_return()

            scores.append(val_score)
            params.append(best_params)

        avg_score = np.mean(scores)
        # Choisir les paramètres du split dont le score est le plus proche de la moyenne
        best_idx = np.argmin(np.abs(np.array(scores) - avg_score))
        best_params = params[best_idx]
        
        return {'avg_score': avg_score, 'cv_scores': scores, 'best_params': best_params}
