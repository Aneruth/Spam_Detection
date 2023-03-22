"""
This file is for parsing and preprocessing the dataset. Sice we have five dataset we can't explicitly 
call the function or implement the method so we are creating a class where ............. (yet to rephrase it)
"""
import logging
import warnings
import typing as t
import re
import string
import os
from nltk.stem import PorterStemmer
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.pipeline import Pipeline
from spam_classifier import __version__ as _version
from spam_classifier.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

warnings.filterwarnings("ignore")


class Parser:
    """
    A class to load the dataset. This class requires pandas package
    """

    def __init__(self):
        self.file_name = 'YoutubeComplete.csv'
        self.col = config.model_config.features
        self.feature = None

        if len(self.col) == 1:
            self.feature = self.col[0]

    def load_dataset(self, *, file_name: str) -> pd.DataFrame:
        """Load the dataset from the path.
        @param file_name: Name of the file to be loaded.
        @return: A pandas dataframe.
        """
        file_name = self.file_name
        if DATASET_DIR is None:
            raise ValueError("Dataset directory not found")
        elif DATASET_DIR == '':
            raise ValueError("Dataset directory value is empty")

        dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
        return dataframe

    def save_pipeline(self, pipeline_to_persist: Pipeline) -> None:
        """Persist the pipeline.
        Saves the versioned model, and overwrites any previous
        saved models. This ensures that when the package is
        published, there is only one trained model that can be
        called, and we know exactly how it was built.

        @param pipeline_to_persist: The pipeline to be persisted.
        @return: None
        """

        # Prepare versioned save file name
        save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
        save_path = TRAINED_MODEL_DIR / save_file_name

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.remove_old_pipelines(files_to_keep=[save_file_name])
        joblib.dump(pipeline_to_persist, save_path)

    def load_pipeline(self, *, file_name: str) -> Pipeline:
        """Load a persisted pipeline.
        @return: A persisted sklearn pipeline.
        """
        file_path = TRAINED_MODEL_DIR / file_name
        trained_model = joblib.load(filename=file_path)
        return trained_model

    def remove_old_pipelines(self, files_to_keep: t.List[str]) -> None:
        """
        Remove old model pipelines.
        This is to ensure there is a simple one-to-one
        mapping between the package version and the model
        version to be imported and used by other applications.
        @param files_to_keep: A list of files to keep in the directory
        @return: None
        """
        do_not_delete = files_to_keep + ["__init__.py"]
        for model_file in TRAINED_MODEL_DIR.iterdir():
            if model_file.name not in do_not_delete:
                model_file.unlink()


class Preprocess:
    """
    A method where we remove all the stopwords,numerics,conjunction,preposition,stemming, and we tokenize the words
    in our dataset.
    This function requires nlp packages such as nltk.corpus, stemmer package (PorterStemmer), Regex, String and
    nltk.tokenize for tokenization.
    """

    def __init__(self):
        self.data = Parser()
        self.dataset = self.data.load_dataset()

    def make_lowercase(self) -> None:
        """Make text lowercase

        @return: None
        """
        self.dataset[self.data.feature] = self.dataset[self.data.feature].str.lower()

    def remove_punctuation(self) -> None:
        """Remove punctuation from the dataset and replace it with empty string
        @return: A pandas series with punctuation removed
        """
        self.dataset[self.data.feature] = self.dataset[self.data.feature].str. \
            translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(self) -> None:
        """Remove stopwords from dataset and rejoins the words without stopwords
        @return: None
        """
        stop_words = set(stopwords.words('english'))
        self.dataset[self.data.feature] = self.dataset[self.data.feature].apply(lambda x: ' '.join(
            [word for word in word_tokenize(x) if word not in stop_words]))

    def remove_numbers(self) -> None:
        """Remove numbers from the dataset and replace it with empty string
        @return: None
        """
        self.dataset[self.data.feature] = self.dataset[self.data.feature].str \
            .replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '')

    def stem_words(self) -> None:
        """Stemming the words in the dataset
        @return: None
        """
        stemmer = PorterStemmer()
        self.dataset[self.data.feature] = self.dataset[self.data.feature].apply(lambda x: ' '.join(
            [stemmer.stem(word) for word in x.split()]))

    def apply(self):
        self.make_lowercase()
        self.remove_punctuation()
        self.remove_stopwords()
        self.remove_numbers()
        self.stem_words()

        print(self.dataset.head(10))

    # def preprocessMethod(self):
    #     # TODO: Split this method into smaller methods
    #
    #     # Stemming
    #     stemmer = PorterStemmer()
    #     stopWordsRemove = [stemmer.stem(i) for i in stopWordsRemove]
    #     dataset.CONTENT = stopWordsRemove
    #     return dataset


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.apply()
