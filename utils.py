import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, List


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



@dataclass
class OptimConfig:
    optimization_period: int  # en jours
    gap_period: int           # en jours
    validation_period: int    # en jours
    n_splits: int

