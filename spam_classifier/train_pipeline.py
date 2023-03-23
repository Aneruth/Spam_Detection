import numpy as np
from config.core import config
from pipeline import CreatePipeline
from Preprocess.preprocess import Parser
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    parser = Parser()
    data = parser.load_dataset()
    pipeline = CreatePipeline(data).create_pipeline()
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    y_train = np.log(y_train)
    # fit model
    pipeline.fit(X_train, y_train)

    # persist trained model
    parser.save_pipeline(pipeline_to_persist=pipeline)


if __name__ == "__main__":
    run_training()
