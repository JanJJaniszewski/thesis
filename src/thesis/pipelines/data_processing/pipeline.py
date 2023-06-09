"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from .dataprep_common import *

from kedro.pipeline import Pipeline, node, pipeline

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
                 name='common_data_prep')
            ]
    )