"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from .dataprep_common import *

from kedro.pipeline import Pipeline, node, pipeline

from .baseline import *
from .noexpert import *
from .expert import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Load and save everything first in the database
            node(func=load_data_and_save,
                 inputs=None,
                 outputs=['raw_frequency', 'raw_severity'],
                 name='load_external_dataset'),
            node(func=common_dataprep,
                 inputs=['raw_frequency', 'raw_severity'],
                 outputs=['freq_train', 'sev_train', 'all_train', 'freq_test', 'sev_test', 'all_test'],
                 name='common_data_prep'),

            # Baseline Regression
            node(func=dataprep_baseline,
                 inputs=['freq_train', 'sev_train'],
                 outputs=None,
                 name='train_lm_baseline_freq'),
            node(func=regression_baseline_freq,
                 inputs=None,
                 outputs=None,
                 name='train_lm_baseline_freq'),
            node(func=regression_baseline_sev,
                 inputs=None,
                 outputs=None,
                 name='train_lm_baseline_sev'),

            # No expert GAN
            node(func=dataprep_noexpert,
                 inputs=['freq_train', 'sev_train'],
                 outputs=['freq_noexpert', 'sev_noexpert'],
                 name='train_lm_baseline_freq'),
            node(func=gan_noexpert_freq,
                 inputs=['freq_noexpert', "params:model_parameters"],
                 outputs='gan_baseline_freq',
                 name='train_gan_baseline_freq'),
            node(func=gan_noexpert_sev,
                 inputs=['sev_noexpert', "params:model_parameters"],
                 outputs='gan_baseline_sev',
                 name='train_gan_baseline_sev'),
            node(func=datagen_noexpert_freq,
                 inputs=['freq_noexpert', "params:model_parameters"],
                 outputs='gan_baseline_freq',
                 name='train_gan_baseline_freq'),
            node(func=datagen_noexpert_sev,
                 inputs=['sev_noexpert', "params:model_parameters"],
                 outputs='gan_baseline_sev',
                 name='train_gan_baseline_sev'),
            node(func=regression_noexpert_freq,
                 inputs=None,
                 outputs=None,
                 name='train_lm_baseline_freq'),
            node(func=regression_noexpert_sev,
                 inputs=None,
                 outputs=None,
                 name='train_lm_baseline_sev'),

            # Expert input GAN
            node(func=dataprep_expert,
                 inputs=['freq_train', 'sev_train'],
                 outputs=None,
                 name='train_lm_baseline_freq'),
            node(func=provide_expert_input,
                 inputs=None,
                 outputs=None,
                 name='train_lm_baseline_freq'),
            node(func=train_gan_expert_freq,
                 inputs=['freq_train', "params:model_parameters"],
                 outputs='gan_baseline_freq',
                 name='train_gan_baseline_freq'),
            node(func=train_gan_expert_sev,
                 inputs=['sev_train', "params:model_parameters"],
                 outputs='gan_baseline_sev',
                 name='train_gan_baseline_sev'),
            node(func=regression_expert_freq,
                 inputs=None,
                 outputs=None,
                 name='train_lm_baseline_freq'),
            node(func=regression_expert_sev,
                 inputs=None,
                 outputs=None,
                 name='train_lm_baseline_sev'),
        ]
    )
