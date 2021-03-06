'''
This file is for parsing and preprocessing the dataset. Sice we have five dataset we can't explicitly 
call the function or implement the method so we are creating a class where ............. (yet to rephrase it)
'''
import warnings
warnings.filterwarnings("ignore")
path = 'YouTubeComplete.csv'
class Parser:    
    '''
    A class to load the dataset. This class requires pandas package
    '''

    def loadData(self):
        """This function considers the path as an input (dataset).

        Returns:
            Dataframe: returns the dataset
        """
        import pandas as pd
        self.dataset = pd.read_csv(path,delimiter=',')
        return self.dataset

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
        self.dataset = data.loadData()[['CONTENT','CLASS']] # choosign only the target columns we need to pass

        # Before applying stop words we need to tokenize the feature column
        # Applying stopwords for our column
        from nltk.corpus import stopwords
        import re,string
        from nltk.tokenize import word_tokenize

        self.stop = stopwords.words('english')
        self.punctiation = set(string.punctuation)
        self.stopWordsRemove = self.dataset['CONTENT'].tolist()
        self.stopWordsRemove = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", c) for c in i.lower().replace("(","").replace(")","").split(' ') if not c.isnumeric()) for i in self.stopWordsRemove]
        self.stopWordsRemove = [' '.join(["".join(j) for j in word_tokenize(i.lower()) if j not in self.stop]) for i in self.stopWordsRemove]
        self.stopWordsRemove = [i for i in self.stopWordsRemove if i not in self.punctiation]

        # Stemming
        from nltk.stem import PorterStemmer
        self.stemmer = PorterStemmer()
        self.stopWordsRemove = [self.stemmer.stem(i) for i in self.stopWordsRemove]
        self.dataset.CONTENT = self.stopWordsRemove
        return self.dataset