import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad

#------------------visualize clinical data------------------#
def cl_vis():
    label = 'sleflare'
    classes = ['SLE inactive', 'SLE active', 'Healthy']
    cl = pd.read_csv('data/clinical_data_clean.csv')
    f = plt.figure(dpi=150, figsize=(6,4))
    counts = cl[label].value_counts().values

    plt.bar(np.arange(3), [76, 41, 87])
    plt.xticks(np.arange(3), classes)
    plt.ylabel('number of samples')
    f.savefig(f'cl_{label}.png')

#------------------clean up expression data------------------#
def data_clean(label='sleflare', outdir='data/normct_all.h5ad'):

    # concatenating expression from different cell types
    #   Progen, MK, pDC were excluded because of diff # samples
    ct_list = ['B', 'cDC', 'cM', 'ncM', 'NK', 'ProlifT', 'Tc', 'Th']
    adata = ad.AnnData()

    for ct in ct_list:
        temp = ad.read_h5ad(f'data/batch_corr/combat_normct_{ct}.h5ad')
        temp.var_names = f'{ct}_cells_' + temp.var_names
        # print(f'{ct} shape is {temp.shape}')
        if ct == 'B':
            adata = temp
            meta = temp.obs
        else:
            adata = ad.concat([adata, temp], axis=1)
    assert adata.shape == (206, 18190 * len(ct_list))

    # data cleanup
    #   change label for healthy patients from nan to 2
    ind_healthy = np.where(meta['disease_cov'] == 'healthy')[0]
    meta.loc[meta.index[ind_healthy], label] = 2

    #   remove sle samples without label information
    mask_sle_noflareinfo = (np.isnan(meta[label])) & (meta['disease_cov']=='sle')
    meta = meta[~mask_sle_noflareinfo]
    adata = adata[~mask_sle_noflareinfo]
    print(meta.shape)
    print(adata.shape)

    # split train and test, add split label
    np.random.seed(0)
    sample_names = meta.copy().index.tolist()
    np.random.shuffle(sample_names)
    meta['split'] = ''
    meta.loc[sample_names[0:144], 'split'] = 'train'
    meta.loc[sample_names[144:174], 'split'] = 'dev'
    meta.loc[sample_names[174:], 'split'] = 'test'

    # write to disk
    adata.obs = meta
    adata.write_h5ad(filename=outdir)

#--------------plot conv1d layer weights---------#
def plot_conv1d_wts():
    ct_list = ['B', 'cDC', 'cM', 'ncM', 'NK', 'ProlifT', 'Tc', 'Th']
    weights = [-0.6229, 0.6325, 0.1358, 0.3103, -0.6641, 0.5187, 0.1810, 0.0885]

    f, ax = plt.subplots()
    plt.style.use('seaborn')
    ax.barh(np.arange(len(ct_list)), weights)
    ax.set_yticks(np.arange(len(ct_list)))
    ax.set_yticklabels(ct_list)
    ax.set_ylim((-0.5, 7.5))
    ax.set_ylabel('cell type')
    ax.set_xlabel('projection layer weight')
    ax.vlines(0, -0.5, 7.5, colors='k')
    plt.show()

if __name__ == "__main__":
    label = 'kidney'
    outdir = 'data/normct_all_kidney.h5ad'
    data_clean(label, outdir)
