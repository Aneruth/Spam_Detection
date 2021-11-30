'''
This file is for parsing and preprocessing the dataset. Sice we have five dataset we can't explicitly 
call the function or implement the method so we are creating a class where ............. (yet to rephrase it)
'''
import warnings
warnings.filterwarnings("ignore")

class Parser:    
    '''
    A class to load the dataset. This class requires pandas package
    '''

    def loadData(self,path):
        '''
        We pass the path of our dataset.
        '''
        import pandas as pd
        self.dataset = pd.read_csv(path)
        # After analysing the dataset we can use two columns that is CONTENT and CLASS
        return self.dataset

class Preprocess:
    '''
    A method where we remove all the stopwords,numerics,conjuntion,preposition,stemming and we tokenize the words 
    in our dataset.
    This function requires nlp packages sunch as nltk.corpus, stemmer package (PorterStemmer), Regex, String and 
    nltk.tokenize for tokenization.
    '''
    def preprocessMethod(self,path):
        '''
        Takes the output from the previos class where the dataset is generated.
        '''
        data = Parser()
        self.dataset = data.loadData(path)[['CONTENT','CLASS']]

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


''' Testing '''
if __name__ == '__main__':

    data = Preprocess()
    # testing
    print(data.preprocessMethod('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Data/Youtube01-Psy.csv'))