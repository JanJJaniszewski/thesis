"""
This is a boilerplate pipeline 'baseline'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .baseline import *
from ..global_functions import *

def create_pipeline(**kwargs) -> Pipeline:
    # Baseline Regression
    return pipeline(
        [
    node(func=dataprep_baseline,
         inputs=['freq_train', 'sev_train'],
         outputs=['freq_train_baseline', 'sev_train_baseline'],
         name='prepare_data_base'),
    node(func=regression_common_freq,
         inputs='freq_train_baseline',
         outputs='regression_baseline_freq',
         name='train_lm_freq_base'),
    node(func=regression_common_sev,
         inputs='sev_train_baseline',
         outputs='regression_baseline_sev',
         name='train_lm_sev_base')
        ]
    )
