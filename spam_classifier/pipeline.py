from sklearn.pipeline import Pipeline
from spam_classifier.Preprocess.preprocess import Parser, Preprocess


class CreatePipeline:
    """ A class to create the pipeline."""

    def __init__(self, dataset):
        self.dataset = dataset

    def create_pipeline(self) -> Pipeline:
        """
        Create the pipeline
        @return: A pipeline object
        """
        preprocess = Preprocess(self.dataset)
        pipeline = Pipeline([
            ('punctuation', preprocess.remove_punctuation()),
            ('lowercase', preprocess.make_lowercase()),
            ('stopwords', preprocess.remove_stopwords()),
            ('numerics', preprocess.remove_numbers()),
            ('lemmatization', preprocess.stem_words())
        ])
        return pipeline
