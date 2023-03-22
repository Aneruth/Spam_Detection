from sklearn.pipeline import Pipeline
from spam_classifier.Preprocess.preprocess import Preprocess


class CreatePipeline:
    """ A class to create the pipeline."""

    def __init__(self):
        self.preprocess = Preprocess()

    def create_pipeline(self) -> Pipeline:
        """
        Create the pipeline
        @return: A pipeline object
        """
        pipeline = Pipeline([
            ('punctuation', self.preprocess.remove_punctuation()),
            ('lowercase', self.preprocess.make_lowercase()),
            ('stopwords', self.preprocess.remove_stopwords()),
            ('numerics', self.preprocess.remove_numbers()),
            ('lemmatization', self.preprocess.stem_words())
        ])
        return pipeline
