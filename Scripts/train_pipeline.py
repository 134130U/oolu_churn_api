import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from Scripts import pipeline, config
from Scripts.data_management import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE,target_file=config.TARGET_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.3, random_state=1, stratify=data[config.TARGET]
    )  # we are setting the seed here
    print(data[config.TARGET].value_counts())

    pipeline.churn_pipe.fit(X_train, y_train)
    print(pipeline.churn_pipe.predict_proba(X_train[config.FEATURES])[:, 1])
    print(pipeline.churn_pipe.predict_proba(X_test[config.FEATURES])[:, 1])

    save_pipeline(pipeline_to_persist=pipeline.churn_pipe)
    print(100*'=')
    print(f'Well done! \n The model is well trained and now saved in the train_modeles folder')
    print(100 * '=')

    test_pred = pipeline.churn_pipe.predict(X_test)
    acc = np.mean(test_pred == y_test)
    print(f'the test accuracy is {acc}')


if __name__ == "__main__":
    run_training()
