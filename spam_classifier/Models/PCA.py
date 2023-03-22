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
        import torch
        
        nrComponets = 100
        # print(type(X_train) == "<class 'scipy.sparse.csr.csr_matrix'>")
        scaler = StandardScaler()
        if torch.is_tensor(X_train): # if the X_train is tensor
            
            """ We are converting the 3d array to 2d array this is because the tensor produces 3 diemensions namely
            [m,n,3] so we transpose and reshape the array to [3,n,m] which yeilds 2d array."""

            X_train = X_train.numpy().transpose().reshape(X_train.shape[0], (X_train.shape[1]*X_train.shape[2]))
            X_test = X_test.numpy().transpose().reshape(X_test.shape[0], (X_test.shape[1]*X_test.shape[2]))
            scaler.fit(X_train)
            X_sc_train = scaler.transform(X_train)
            X_sc_test = scaler.transform(X_test)
            pca = PCA(n_components = X_train.shape[0]) # To total number of rows present
            pca.fit(X_train)
            pca = PCA(n_components=nrComponets)
            X_pca_train = pca.fit_transform(X_sc_train)
            X_pca_test = pca.transform(X_sc_test)
            pca_std = np.std(X_pca_train)
        else:
            scaler.fit(X_train.toarray())
            X_sc_train = scaler.transform(X_train.toarray())
            X_sc_test = scaler.transform(X_test.toarray())
            pca = PCA(n_components = X_train.shape[0]) # To total number of rows present
            pca.fit(X_train.toarray())
            pca = PCA(n_components=nrComponets)
            X_pca_train = pca.fit_transform(X_sc_train)
            X_pca_test = pca.transform(X_sc_test)
            pca_std = np.std(X_pca_train)

        '''plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')'''

        print(f'Shape of X_train before PCA {X_sc_train.shape}')
        print(f'Shape of X_train after PCA {X_pca_train.shape}')

        # reshape our labels as 2d-tensor (the first dimension will be the batch dimension and the second the scalar label)
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))

        return pca_std,X_pca_test,X_pca_train, y_test,y_train