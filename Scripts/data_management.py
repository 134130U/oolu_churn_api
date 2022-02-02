import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from Scripts import config

import logging
import typing as t


_logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str, target_file: str) -> pd.DataFrame:
    feat = pd.read_csv(f"{config.DATASET_DIR}/{file_name}", low_memory=False)
    targ = pd.read_csv(f"{config.DATASET_DIR}/{target_file}", low_memory=False)
    _data = pd.merge(feat, targ, on='account_id')
    _data = _data[_data['status'] == 1]
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    api_path = "/home/aims/Oolu/myapi/trained_models/" + save_file_name

    # remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    # joblib.dump(pipeline_to_persist, api_path)
    # _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    with open(file_path, 'rb') as f:
        trained_model = joblib.load(f)
    return trained_model


# def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
#     """
#     Remove old model pipelines.
#     This is to ensure there is a simple one-to-one
#     mapping between the package version and the model
#     version to be imported and used by other applications.
#     However, we do also include the immediate previous
#     pipeline version for differential testing purposes.
#     """
#     do_not_delete = files_to_keep + ['__init__.py']
#     for model_file in config.TRAINED_MODEL_DIR.iterdir():
#         if model_file.name not in do_not_delete:
#             model_file.unlink()