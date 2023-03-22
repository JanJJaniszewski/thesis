"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines import data_processing as common
from .pipelines import baseline as base
from .pipelines import no_expert_input as no_expert_input
from .pipelines import expert_input as expert_input


# def register_pipelines() -> Dict[str, Pipeline]:
#     """Register the project's pipelines.
#
#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     print(pipelines)
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    commonp = common.create_pipeline()
    basep = base.create_pipeline()
    nep = no_expert_input.create_pipeline()
    ep = expert_input.create_pipeline()

    return {
        "__default__": commonp + basep + nep + ep,
        'common': commonp,
        'baseline': basep,
        'no_expert_input': nep,
        'expert_input': ep
    }
