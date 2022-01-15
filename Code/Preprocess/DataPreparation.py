import sys
sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Preprocess')

import warnings
warnings.filterwarnings("ignore")

from preprocess import Preprocess

class dataPrepare:
    '''
    A class where we vecorise the dataset with the help of tfidf vectorizer and then we use the traditional traintest split method for generating the
    data for machine learning model.
    '''

    def Vectorizer(self,path):
        """The term frequency–inverse document frequency, is a numerical statistic that is intended to reflect 
        how important a word is to a document in a collection or corpus.
        We are using this method to convert our text data to a sparse matrix (dense).

        Args:
            path (String): Path of our daatset

        Returns:
            DataFrame: returns the input for our algorithms(DataFrame/DataSeries)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")

        self.data = Preprocess()
        self.data = self.data.preprocessMethod(path)
        
        self.feature = vec.fit_transform(self.data.CONTENT)
        
        # Printing the shape of our features
        # print(f'Shape of our fearture is {self.feature.shape}')

        # Train test split our data
        from sklearn.model_selection import train_test_split

        self.X = self.feature
        self.y = self.data.CLASS
        
        # Spliting our dataset into 30% for testing and 70% for training
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.3)

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def deepLearningInput(self,path):
        """Input to for our deep learning algorithms

        Args:
            path (String): Path of our daatset

        Returns:
            DataFrame: returns the input for our algorithms(DataFrame/DataSeries)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")

        self.data = Preprocess()
        self.data = self.data.preprocessMethod(path)
        
        self.feature = vec.fit_transform(self.data.CONTENT)

        self.X = self.feature
        self.y = self.data.CLASS

        return self.X,self.y
