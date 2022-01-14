import sys
# sys.path.insert(0, '/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Preprocess')
sys.path.append("../") 
from DataPreparation import dataPrepare

class PrincipleComponentAnalysis:
    
    def __init__(self,path):
        """PCA is a method for lowering the dimensionality for specific datasets, boosting interpretability while 
        minimizing information loss. It accomplishes this by generating new uncorrelated variables 
        that optimize variance in a sequential manner.

        Args:
            path (String): Path of our dataset
        Returns:
            DataFrame: returns PCA input for the Neural Network and LSTM algorithms (DataFrame/DataSeries)
        """
        self.path = path 
    
    def produceData(self,X_train, X_test, y_train, y_test):
        """Considers the input from the fetchData module and returns the dataset with lower diemensions.

        Returns:
            - numpy.float64: PCA standard devision for Gaussian Noise
            - ndarray: X_test for PCA
            - Pandas Series: y_test
            - ndarray: y_train
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import numpy as np

        scaler = StandardScaler()
        scaler.fit(X_train.toarray())
        X_sc_train = scaler.transform(X_train.toarray())
        X_sc_test = scaler.transform(X_test.toarray())

        pca = PCA(n_components = X_train.shape[0]) # To total number of rows present
        pca.fit(X_train.toarray())

        '''plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')'''

        nrComponets = 100
        pca = PCA(n_components=nrComponets)
        X_pca_train = pca.fit_transform(X_sc_train)
        X_pca_test = pca.transform(X_sc_test)
        pca_std = np.std(X_pca_train)

        print(f'Shape of X_train before PCA {X_sc_train.shape}')
        print(f'Shape of X_train after PCA {X_pca_train.shape}')

        # reshape our labels as 2d-tensor (the first dimension will be the batch dimension and the second the scalar label)
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))

        return pca_std,X_pca_test,X_pca_train, y_test,y_train

if __name__ == '__main__':
    pca = PrincipleComponentAnalysis('/Users/aneruthmohanasundaram/Documents/GitHub/Spam_Detection/Code/Data/Youtube01-Psy.csv')
    a,b,c,d = pca.produceData()
    print(type(a),type(b),type(c),type(d))