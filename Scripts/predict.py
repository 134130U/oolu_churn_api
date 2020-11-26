from Scripts import config
import joblib


# pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}.pkl"
churn_pipe = joblib.load(config.PACKAGE_ROOT / 'oolu_churn_model.pkl')


def make_prediction(input_data):
    """Make a prediction using a saved model pipeline.
    Args:
        input_data: Array of model prediction inputs.
    Returns:
        Predictions for each input row, as well as the model version.
    """

    validated_data = input_data.copy()

    return churn_pipe.predict_proba(validated_data[config.FEATURES])[:, 1]*100