from datetime import datetime, timedelta
from typing import Tuple, List

from utils import OptimConfig



class Sampling:
    
    def __init__(self, start: datetime, end: datetime, optim_config):
        self.start_date = start
        self.end_date = end
        self.optim_config = optim_config
        
    def generate_splits(self) -> List[Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]]:
        splits = []
        period_length = (
            self.optim_config.optimization_period +
            self.optim_config.gap_period +
            self.optim_config.validation_period
        )
        
        if self.optim_config.n_splits == 1:
            # Pour un seul split, on commence au début
            optim_start = self.start_date
            optim_end = optim_start + timedelta(days=self.optim_config.optimization_period)
            valid_start = optim_end + timedelta(days=self.optim_config.gap_period)
            valid_end = valid_start + timedelta(days=self.optim_config.validation_period)
            
            if valid_end <= self.end_date:
                splits.append(((optim_start, optim_end), (valid_start, valid_end)))
        else:
            # Pour plusieurs splits
            total_days = (self.end_date - self.start_date).days
            step = (total_days - period_length) // (self.optim_config.n_splits - 1) if self.optim_config.n_splits > 1 else 0
            
            for i in range(self.optim_config.n_splits):
                optim_start = self.start_date + timedelta(days=i * step)
                optim_end = optim_start + timedelta(days=self.optim_config.optimization_period)
                valid_start = optim_end + timedelta(days=self.optim_config.gap_period)
                valid_end = valid_start + timedelta(days=self.optim_config.validation_period)
                
                if valid_end <= self.end_date:
                    splits.append(((optim_start, optim_end), (valid_start, valid_end)))
        
        return splits
    


