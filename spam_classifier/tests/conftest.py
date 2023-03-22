import pytest

from spam_classifier.config.core import config
from spam_classifier.Preprocess.preprocess import Parser


@pytest.fixture()
def sample_input_data():
    parse = Parser()
    return parse.load_dataset(file_name=config.app_config.test_data_file)