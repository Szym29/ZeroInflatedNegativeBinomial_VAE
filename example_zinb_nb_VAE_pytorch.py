import numpy as np
import scanpy as sc
from utils import H5adDataSet
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import scipy.sparse as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from torch.utils.data import DataLoader
from torch.distributions import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from scipy.stats import spearmanr
def _cosine_similarity(y_true, y_pred):
    # Normalize the rows of A and B
    true_norm = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True)+ 0.00001)
    pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True)+ 0.00001)

    # Compute the cosine similarity
    cosine_similarity = np.sum(true_norm * pred_norm, axis=1).mean()

    return cosine_similarity

def _spearman_r(y_true, y_pred):
    corrs = np.array([spearmanr(y_true[i], y_pred[i])[0] for i in range(y_pred.shape[0])])
    return corrs.mean()


def metrics(y_true, y_pred):
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    scores = {
        "val/mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "val/mean_squared_error": mean_squared_error(y_true, y_pred),
        "val/cosine_similarity": _cosine_similarity(y_true, y_pred),
        "val/r2_score": r2_score(y_true, y_pred),
        "val/l0_diff": np.abs(np.sum(y_true != 0)-np.sum(y_pred !=0)),
       "val/spearmanr": _spearman_r(y_true, y_pred)
    }
    return scores
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,distribution='zinb',log_data = False):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        self.distribution = distribution
        self.log_data = log_data
        self.log_theta = torch.nn.Parameter(torch.randn(input_dim))
    def encode(self, x):
        if not self.log_data:
            x = torch.log(x+1)
        h1 = F.softplus(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.softplus(self.fc3(z))
        mu = self.fc4(h3)
        dropout_logits = self.fc5(h3)
        return torch.exp(mu), dropout_logits

    def forward(self, x):   
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        #zinb distribution
        de_mean, de_dropout = self.decode(z)
        return de_mean, de_dropout, mu, logvar
    

    def get_latent_representation(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        return mu + torch.exp(0.5 * logvar)
    # def getnerate
    def generate(self, x, sample_shape,random=False):
        '''
        generate samples from the model
        sample_shape: shape of sample
        '''
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        if random:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu + torch.exp(0.5 * logvar) #not using reparameterize
        mu, dropout_logits = self.decode(z)
        theta = self.log_theta.exp()
        nb_logits = (mu+1e-4).log() - (theta+1e-4).log()
        if self.distribution == 'zinb':
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
        elif self.distribution == 'nb':

            distribution = NegativeBinomial(total_count=theta, logits=nb_logits,validate_args=False)
        if random:
            return distribution.sample(sample_shape) #return the sample of zinb distribution
        else:
            return distribution.mean #return the mean of zinb distribution
    def kl_d(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=-1),dim=0)
    def reconstruction_loss(self, x, mu, dropout_logits):
        '''
        x: input data
        mu: output of decoder
        dropout_logits: dropout logits of zinb distribution
        '''
        theta = self.log_theta.exp()
        
        nb_logits = (mu+1e-5).log() - (theta+1e-5).log()
        
        if self.distribution == 'zinb':
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits,gate_logits = dropout_logits, validate_args=False)
        elif self.distribution == 'nb':
            distribution = NegativeBinomial(total_count=theta, logits=nb_logits,validate_args=False)
        return distribution.log_prob(x).sum(-1).mean()
    def loss_function(self, x, mu, dropout_logits, mu_, logvar_):
        reconstruction_loss = self.reconstruction_loss(x, mu, dropout_logits)
        kl_div = self.kl_d(mu_, logvar_)
        return -reconstruction_loss + kl_div



def preprocess(adata):
    pass

def _cosine_similarity(y_true, y_pred):
    # Normalize the rows of A and B
    true_norm = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True)+ 0.00001)
    pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True)+ 0.00001)

    # Compute the cosine similarity
    cosine_similarity = np.sum(true_norm * pred_norm, axis=1).mean()

    return cosine_similarity

def main(data_directory, distribution='zinb',plot_embedding=False,clustering=False,label_name = None, lr=1.0e-4, use_cuda=False, num_epochs=10, batch_size=10,left_trim=False,output='output',log_data = False):
    adata = sc.read(data_directory)
    print('Using distribution: ', distribution)
    data_name = data_directory.split('/')[-1].split('.')[0]
    if left_trim:
        adata.X[adata.X<0]=0
    # adata = preprocess(adata) if the data need to be preprocessed
    assert np.min(adata.X) >= 0, 'Your data has negative values, pleaes specify --left_trim True if you still want to use this data'
    dataset = H5adDataSet(adata)
    cell_loader = DataLoader(dataset,batch_size=batch_size)
    vae = VAE(input_dim = dataset.num_genes(), hidden_dim = 400, latent_dim=40,distribution=distribution,log_data=log_data) #distribution = 'nb' to use negative binomial distribution
    if use_cuda:
        vae.cuda()
    optimizer = optim.Adam(lr=lr, params=vae.parameters())
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(cell_loader):
            if use_cuda:
                data = data.cuda()
            optimizer.zero_grad()
            mu, dropout_logits, mu_, logvar_ = vae(data)
            loss = vae.loss_function(data, mu, dropout_logits, mu_, logvar_)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(cell_loader.dataset)))
    if not os.path.exists(output):
        os.makedirs(output)
    
    TZ=[]
    preds = []
    for x in cell_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        z=vae.get_latent_representation(x)
        # pred = vae.generate(x, sample_shape=1,random=False)
        if use_cuda:
            zz=z.cpu().detach().numpy().tolist()
            # pred = pred.cpu().detach().numpy().tolist()
        else:
            zz=z.detach().numpy().tolist()
            # pred = pred.detach().numpy().tolist()
        TZ+=zz
        # preds+=pred
        print(len(preds))
    TZ = np.array(TZ)
    adata.obsm['embeddings'] = TZ
    # adata.layers['pred'] = np.array(preds)
    sc.pp.neighbors(adata, use_rep='z')
    if plot_embedding:
        sc.tl.umap(adata)
        if label_name is not None:
            sc.pl.umap(adata,color=label_name,size=10,save='_{}_{}_{}_embedding.png'.format(data_name,distribution,label_name))
        if clustering:
            sc.tl.leiden(adata)
            sc.pl.umap(adata,color='leiden',size=10, save='_{}_{}_leiden_clustering.png'.format(data_name,distribution))
    print(metrics(adata.X, np.array(preds)))
    # torch.save(vae, output) #save the model if you want
    adata.write('output.h5ad')
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None, help='data directory')
parser.add_argument('--plot_embedding', type=bool, default=True, help='plot latent space embedding')
parser.add_argument('--clustering', type=bool, default=True, help='do leiden clustering')
parser.add_argument('--label_name', type=str, default=None, help='the name of ground truth label if applicable')
parser.add_argument('--lr', type=float, default=1.0e-5, help='the learning rate of the model')
parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda or not')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--left_trim', type=bool, default=False, help='if the data has negative values, please specify True')
parser.add_argument('--output', type=str, default='output', help='the output directory of the model and plots')
parser.add_argument('--distribution', type=str, default='zinb', help='one distribution of [zinb,nb]')
parser.add_argument('--log_data', type=bool, default=False, help='if the data is already log transformed, specify True. Default is False')
args = parser.parse_args()
if __name__ == '__main__':

    assert args.data_dir != None,'Please provide the data directory!'
    main(args.data_dir,args.distribution, args.plot_embedding,args.clustering,args.label_name,args.lr,args.use_cuda,args.num_epochs,args.batch_size,args.left_trim,args.output,args.log_data)
