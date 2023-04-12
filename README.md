# ZeroInflatedNegativeBinomial_VAE
This is a very fundamental Variational AutoEncoder model with zero inflated negative binomial distribution and negative binomial distribution. It can be widely applied in the single cell RNA (scRNA) or single cell ATAC (scATAC) data to extract compat low dimensionality representations. (Other data distribution might be added to this repo later on.) 

For beginers who are inetrested in using VAE for single cell analysis, you can use this as an toy example to play with different parameters like batch_size, learning rate to see how to achieve a better low dimensionality represetnations. 

For advanced users or who want to customize their own VAE models on single cell data,  you can easily build a customized VAE model by changing the number of layers and number of neurons in the encoder/decoder structure by specifying the key arguments, or  adding the novel deep learning components, customizing the loss function, and etc.

## Dependencies

- Python 3
- pytorch
- pyro
- anndata
- scanpy

## Datasets

Example data `example.h5ad` is provided to testing the toy example only. The cell type label is `celltype` in the `uns` attribute. 

## Usage

```
git clone https://github.com/Szym29/ZeroInflatedNegativeBinomial_VAE.git
cd ZeroInflatedNegativeBinomial_VAE
```

Please use `python example_zinb_nb_VAE_pytorch.py -h` to find the usage of keywords like input data directory, leraning rate, assumption of data distribution, and the name of cell type attributes, plotting latent embedding with umaps or not, and etc. 

The `--data_dir` is required for running the script

Example:

```
python example_zinb_nb_VAE_pytorch.py --data_dir example.h5ad --distribution zinb --use_cuda True --plot_embedding True --clustering True --lable_name celltype
```

The VAE model will apply zero-inflated negative binomial distribution and be trained using GPU, and doing `leiden` clustering on the latent representation level. Plot the latent representation using umap and coloring by cell types labels and clustering results. (Plots can be found at `./figures/`)

If your single cell data is not preprocessed, you can extends the `preprocess(adata)` function and uncomment the `adata = preprocess(adata)` in the `main` function to use the preprocessed data. You can also simply pass the directory of preprocessed data into the script. 

**NOTE**: if the preprocessed data or your data contains negative values, the script will stop running because this violate the assumption of both negative binomial and zero-inflated negative binomial distribution. Druing the training, the loss will be NaN. If you still want to use the the data in training a VAE model, please specify `--left_trim True` when you run the python script. This keyword will force all negative values as 0 and the model will run without bugs. But this strategy will loss some information from data.

## Contact

[Yumin Zheng](mailto:zhengyumin529@gmail.com)
