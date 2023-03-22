'''
This file is for parsing and preprocessing the dataset. Sice we have five dataset we can't explicitly 
call the function or implement the method so we are creating a class where ............. (yet to rephrase it)
'''
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

    def load_dataset(self, *, file_name: str) -> pd.DataFrame:
        dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
        dataframe["MSSubClass"] = dataframe["MSSubClass"].astype("O")

        # rename variables beginning with numbers to avoid syntax errors later
        transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
        return transformed

class Preprocess:
    '''
    A method where we remove all the stopwords,numerics,conjuntion,preposition,stemming and we tokenize the words 
    in our dataset.
    This function requires nlp packages sunch as nltk.corpus, stemmer package (PorterStemmer), Regex, String and 
    nltk.tokenize for tokenization.
    '''
    def preprocessMethod(self):
        """Takes the output from the previous class where the dataset is generated.

        Args:
            path (String): Path of the dataset

        Returns:
            Dataframe: Return the dataset(target columns) where we clean and preprocess it.
        """
        data = Parser()
        dataset = data.loadData()[['CONTENT','CLASS']] # choosign only the target columns we need to pass

        # Before applying stop words we need to tokenize the feature column
        # Applying stopwords for our column
        from nltk.corpus import stopwords
        import re,string
        from nltk.tokenize import word_tokenize

        stop = stopwords.words('english')
        punctiation = set(string.punctuation)
        stopWordsRemove = dataset['CONTENT'].tolist()
        stopWordsRemove = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", c) for c in i.lower().replace("(","").replace(")","").split(' ') if not c.isnumeric()) for i in stopWordsRemove]
        stopWordsRemove = [' '.join(["".join(j) for j in word_tokenize(i.lower()) if j not in stop]) for i in stopWordsRemove]
        stopWordsRemove = [i for i in stopWordsRemove if i not in punctiation]

        # Stemming
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        stopWordsRemove = [stemmer.stem(i) for i in stopWordsRemove]
        dataset.CONTENT = stopWordsRemove
        return dataset