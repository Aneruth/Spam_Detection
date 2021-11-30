import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

def loadData(dataset):
    '''
    This function preprocess the data and performs analysis for all the dataset and returns the dataset.
    '''

    df = pd.read_csv(dataset)
    # print the dataset
    print(df)

    # Data Analysis --> EDA
    # To check the labels(Class)
    print("\n",df.groupby('CLASS').describe().T)

    # Creating a new column to set the length of our content column
    df['comment_len'] = df.CONTENT.apply(len)

    print("\n",df.CONTENT.value_counts().rename_axis(['CONTENT']).reset_index(name='counts').head())

    df.CLASS.value_counts().plot(kind = 'pie',explode=[0, 0.1],figsize=(6, 6),autopct='%1.1f%%',shadow=True)
    plt.title("Spam vs No Spam")
    plt.legend(["No Spam", "Spam"])
    plt.show()

    plt.figure(figsize=(12,6))
    df.comment_len.plot(bins=100, kind='hist') # with 100 length bins (100 length intervals) 
    plt.title("Frequency Distribution of Message Length")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()

    df.hist(column='comment_len', by='CLASS', bins=50,figsize=(12,4))
    plt.show()

    # Applying Stopwords 
    stop = stopwords.words('english')

    # Cretaing a new column after removing the stopwords and converting it to lower case
    df['Content_without_stopwords'] = df.CONTENT.apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in (stop)]))

    # Removing the puntuations
    tokenizer = RegexpTokenizer(r'\w+')
    df['tokenize_column'] = [tokenizer.tokenize(i) for i in df.Content_without_stopwords]

    # Stemming the words# We have a various stemmer option and  don't know what to choose. 
    # https://towardsdatascience.com/stemming-corpus-with-nltk-7a6a6d02d3e5


    stemmer = SnowballStemmer(language='english')
    # [stemmer.stem(j) for i in df.loc[0:2]['tokenize_column'].tolist() for j in i]    
    
    return df

