import pathlib
import trained_models
import datasets, Query

PACKAGE_ROOT = pathlib.Path(trained_models.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT
DATASET_DIR = pathlib.Path(datasets.__file__).resolve().parent
QUERY_DIR = pathlib.Path(Query.__file__).resolve().parent
TARGET_FILE = "target_2_21.csv"
TRAINING_DATA_FILE = "churndata.csv"
query = QUERY_DIR / 'finaleQuery.sql'
jsn = QUERY_DIR / 'json_script.sql'

ID = ['account_id', 'slug']

TARGET = 'churn'
# variables
FEATURES = ['total_amount', 'day_disabled', 'expected_total_amount', 'created_at',
            'total_payed', 'status', 'cutoff_days', 'prev_payment']
print(DATASET_DIR)

PIPELINE_NAME = "oolu_churn"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_model"
