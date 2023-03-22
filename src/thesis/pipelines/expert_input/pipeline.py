"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from ..global_functions import *
from .expert import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=dataprep_expert,
                 inputs=['freq_train', 'sev_train'],
                 outputs=['freq_train_expert', 'sev_train_expert'],
                 name='prepare_data_e'),

            node(func=train_expert_models,
                 inputs=['freq_train_expert', 'sev_train_expert'],
                 outputs=['expertmodel_freq', 'expertmodel_sev'],
                 name='train_expert_models_e'),
            node(func=provide_expert_input,
                 inputs=['freq_train_expert', 'sev_train_expert', 'expertmodel_freq', 'expertmodel_sev'],
                 outputs=['freq_train_expert_input', 'sev_train_expert_input'],
                 name='provide_expert_input_e'),

            node(func=train_gan_expert_freq,
                 inputs=['freq_train_expert_input', "params:model_parameters"],
                 outputs='gan_expert_freq',
                 name='train_freq_gan_e'),
            node(func=train_gan_expert_sev,
                 inputs=['sev_train_expert_input', "params:model_parameters"],
                 outputs='gan_expert_sev',
                 name='train_sev_gan_e'),
            node(func=generate_gan,
                 inputs=['gan_expert_freq', "freq_train_expert_input"],
                 outputs='generated_expert_train_freq',
                 name='generate_freq_data_e'),
            node(func=generate_gan,
                 inputs=['gan_expert_sev', "sev_train_expert_input"],
                 outputs='generated_expert_train_sev',
                 name='generate_sev_data_e'),

            node(func=regression_common_freq,
                 inputs='generated_expert_train_freq',
                 outputs='regression_expert_freq',
                 name='train_lm_freq_e'),
            node(func=regression_common_sev,
                 inputs='generated_expert_train_sev',
                 outputs='regression_expert_sev',
                 name='train_lm_sev_e')
        ]
    )
