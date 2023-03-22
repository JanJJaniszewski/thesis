"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .noexpert import *
from ..global_functions import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=dataprep_noexpert,
                 inputs=['freq_train', 'sev_train'],
                 outputs=['freq_train_noexpert', 'sev_train_noexpert'],
                 name='prepare_data_ne'),

            # As baseline but with those four nodes added
            node(func=train_gan_noexpert_freq,
                 inputs=['freq_train_noexpert', "params:model_parameters"],
                 outputs='gan_noexpert_freq',
                 name='train_sev_gan_ne'),
            node(func=train_gan_noexpert_sev,
                 inputs=['sev_train_noexpert', "params:model_parameters"],
                 outputs='gan_noexpert_sev',
                 name='train_freq_gan_ne'),
            node(func=generate_gan,
                 inputs=['gan_noexpert_freq', "freq_train_noexpert"],
                 outputs='generated_noexpert_train_freq',
                 name='generate_freq_data_ne'),
            node(func=generate_gan,
                 inputs=['gan_noexpert_sev', "sev_train_noexpert"],
                 outputs='generated_noexpert_train_sev',
                 name='generate_sev_data_ne'),

            # Common to baseline
            node(func=regression_common_freq,
                 inputs='generated_noexpert_train_freq',
                 outputs='regression_noexpert_freq',
                 name='train_lm_freq_ne'),
            node(func=regression_common_sev,
                 inputs='generated_noexpert_train_sev',
                 outputs='regression_noexpert_sev',
                 name='train_lm_sev_ne')
        ]
    )
