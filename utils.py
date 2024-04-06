from torch.utils.data import Dataset
from scipy.sparse import csc_matrix
import numpy as np
import torch
class H5adDataSet(Dataset):
    '''
    The customized dataset for the data without gene weights. (For the initial iteration) The dataset will return the gene expression and the cell graph for each cell.
    '''
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return self.data.X.shape[0]
    def __getitem__(self,idx):

        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32)
        x_tensor=torch.from_numpy(x)
        return x_tensor
    def num_genes(self):
        return len(self.data.var)
    def num_cells(self):
        return len(self.data.obs)
