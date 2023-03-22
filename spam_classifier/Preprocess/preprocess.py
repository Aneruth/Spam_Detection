"""
This file is for parsing and preprocessing the dataset. Sice we have five dataset we can't explicitly 
call the function or implement the method so we are creating a class where ............. (yet to rephrase it)
"""
import warnings
import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from spam_classifier import __version__ as _version
from spam_classifier.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

warnings.filterwarnings("ignore")


class Parser:
    """
    A class to load the dataset. This class requires pandas package
    """

    def __init__(self, file_name: str):
        self.file_name = file_name

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from the path."""
        dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{self.file_name}"))
        return dataframe

    def save_pipeline(self, pipeline_to_persist: Pipeline) -> None:
        """Persist the pipeline.
        Saves the versioned model, and overwrites any previous
        saved models. This ensures that when the package is
        published, there is only one trained model that can be
        called, and we know exactly how it was built.
        """

        # Prepare versioned save file name
        save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
        save_path = TRAINED_MODEL_DIR / save_file_name

        self.remove_old_pipelines(files_to_keep=[save_file_name])
        joblib.dump(pipeline_to_persist, save_path)

    def load_pipeline(self) -> Pipeline:
        """Load a persisted pipeline."""

        file_path = TRAINED_MODEL_DIR / self.file_name
        trained_model = joblib.load(filename=file_path)
        return trained_model

    def remove_old_pipelines(self, files_to_keep: t.List[str]) -> None:
        """
        Remove old model pipelines.
        This is to ensure there is a simple one-to-one
        mapping between the package version and the model
        version to be imported and used by other applications.
        @param files_to_keep: 
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

    def preprocessMethod(self):
        # TODO: Split this method into smaller methods
        """Takes the output from the previous class where the dataset is generated.

        Returns:
            Dataframe: Return the dataset(target columns) where we clean and preprocess it.
        """
        data = Parser()
        dataset = data.loadData()[['CONTENT', 'CLASS']]  # choosing only the target columns we need to pass

        # Before applying stop words we need to tokenize the feature column
        # Applying stopwords for our column
        from nltk.corpus import stopwords
        import re, string
        from nltk.tokenize import word_tokenize

        stop = stopwords.words('english')
        punctiation = set(string.punctuation)
        stopWordsRemove = dataset['CONTENT'].tolist()
        stopWordsRemove = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", c) for c in
                                    i.lower().replace("(", "").replace(")", "").split(' ') if not c.isnumeric()) for i
                           in stopWordsRemove]
        stopWordsRemove = [' '.join(["".join(j) for j in word_tokenize(i.lower()) if j not in stop]) for i in
                           stopWordsRemove]
        stopWordsRemove = [i for i in stopWordsRemove if i not in punctiation]

        # Stemming
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        stopWordsRemove = [stemmer.stem(i) for i in stopWordsRemove]
        dataset.CONTENT = stopWordsRemove
        return dataset
