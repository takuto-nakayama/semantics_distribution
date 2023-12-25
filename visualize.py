from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from plotly import express as px

import pandas as pd
import numpy as np

def mds(df_JSdiv:pd.DataFrame, languages:list, num_file:int):
    # 3-dimentional plotting & the dataframe of the coordinates
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=0).fit_transform(df_JSdiv)
    df_mds = pd.DataFrame(mds, columns=['dim1', 'dim2', 'dim3'], index=df_JSdiv.index)
    df_mds['language'] = [l for l in languages for n in range(num_file)]
    
    fig = px.scatter_3d(df_mds, x='dim1', y='dim2', z='dim3', color='language')
    
    return fig, df_mds



def tsne(df_tagcnt:pd.DataFrame,languages:list, num_file:int, perplexity:int=1):
    df_tagcnt_std = df_tagcnt.apply(lambda x: x/x.sum()).T

    list_sorted = []
    for idx in df_tagcnt_std.index:
        list_sorted.append([i for i in df_tagcnt_std.loc[idx].sort_values(ascending=False)])
    df_tagcnt_std = pd.DataFrame(list_sorted)

    tsne = TSNE(n_components=3, random_state = 30, perplexity = perplexity, n_iter = 1000)
    np_tsne = tsne.fit_transform(df_tagcnt_std)

    df_tsne = pd.DataFrame(np_tsne, index=df_tagcnt.columns , columns=['dim1', 'dim2', 'dim3'])
    df_tsne['language'] = [l for l in languages for n in range(num_file)]
    fig = px.scatter_3d(df_tsne, x='dim1', y='dim2', z='dim3', color='language')

    return fig, df_tsne