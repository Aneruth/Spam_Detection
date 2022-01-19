import sys
import os
sys.path.insert(0, os.path.join('Code/Preprocess'))


class ExploratoryDataAnalysis:
    """A class for exploratory data analysis of our dataset where we summarize the main characteristics of our dataset
    and also use statistical graphics and other data visualization methods for our dataset.
    """
    @staticmethod
    def visualise():
        """A method where we perform the data analysis for our dataset
        """
        from preprocess import Preprocess
        from pathlib import Path
        import seaborn as sns
        import matplotlib.pyplot as plt
        from collections import Counter
        import pandas as pd
        from wordcloud import WordCloud

        data = Preprocess()
        data = data.preprocessMethod('YoutubeComplete.csv')
        data.replace({0.0: 'Ham', 1.0: 'Spam'}, inplace=True)
        data['length'] = data.CONTENT.apply(len)

        immedDir = Path(__file__).parent.parent
        path_to_save = os.path.join(immedDir, "Images/Eda")

        # Check if the output folder path present if not create it
        if not os.path.exists(os.path.join(immedDir, "Images")):
            os.mkdir(os.path.join(immedDir, "Images"))

        if not os.path.exists(os.path.join(immedDir, "Images/Eda")):
            os.mkdir(os.path.join(immedDir, "Images/Eda"))

        fig = data.hist(column='length', by='CLASS', bins=50, figsize=(11, 5))
        plt.savefig(os.path.join(
            path_to_save, "Content Length Distribution"+".png"))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        f, ax = plt.subplots(1, 2, figsize=(12, 4))
        data.CLASS.value_counts().plot.pie(
            explode=[0, 0.12], autopct='%1.3f%%', ax=ax[0])
        ax[0].set_title('Class Ratio')
        sns.countplot('CLASS', data=data)
        ax[1].set_title('Class Distribution')
        plt.savefig(os.path.join(
            path_to_save, "Class Distribution and Ratio"+".png"))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        hamWords = list(data.loc[data.CLASS == 'Ham', 'CONTENT'])
        text_wordcloud_ham = " ".join(x for x in hamWords)
        wordcloud = WordCloud(
            background_color="white").generate(text_wordcloud_ham)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title("Frequently Used Ham Words")
        plt.axis("off")
        plt.savefig(os.path.join(
            path_to_save, "Frequently Used Ham Words"+".png"))
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        spamWords = list(data.loc[data.CLASS == 'Spam', 'CONTENT'])
        text_wordcloud_spam = " ".join(x for x in spamWords)
        wordcloud = WordCloud(
            background_color="white").generate(text_wordcloud_spam)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title("Frequently Used Spam Words")
        plt.axis("off")
        plt.savefig(os.path.join(
            path_to_save, "Frequently Used Spam Words"+".png"))
        plt.show(block=False)
        plt.pause(3)
        plt.close()


if __name__ == '__main__':
    ExploratoryDataAnalysis.visualise()
