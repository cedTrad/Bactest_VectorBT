import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, List

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



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



def add_line(fig, data, feature, name, color = None, col = None, row = None, add_hover = True, dash = None):
    if add_hover:
        fig.add_trace(
            go.Scatter(x = data.index, y = data[feature],
                       name = name,
                       line = dict(color = color, dash = dash)),
            col = col, row = row)
    else:
        fig.add_trace(
            go.Scatter(x = data.index, y = data[feature],
                       name = name, hoverinfo='none'),
            col = col, row = row)
        

def add_scatter(fig, data, feature, name, color = None, size = None, col = None, row = None):
    fig.add_trace(
        go.Scatter(
            x = data.index,
            y = data[feature],
            mode = 'markers',
            marker_color = color,
            marker_size = size,
            name = name
        ),
        col = col, row = row
    )