from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from Scripts import prepare_data as pp, config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

churn_pipe = Pipeline(
    [
        (
            "Preprocess the data",
            pp.Preprocess(select_features=config.FEATURES),
        ),
        ("scaler", MinMaxScaler()),
        ("clf_model", LogisticRegression()),
    ]
)