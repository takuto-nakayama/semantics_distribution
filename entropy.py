import pandas as pd
import math


def entropy(df_tagcnt:pd.DataFrame, languages:list, num_file:int):

    df_tagcnt_std = df_tagcnt.apply(lambda x: x/x.sum())

    matrix_H = []
    list_H = []

    for col in df_tagcnt_std.columns:
        if len(list_H) == num_file-1:
            list_H.append(sum([-df_tagcnt_std[col][idx] * math.log(df_tagcnt_std[col][idx], 2) for idx in range(len(df_tagcnt_std[col])) if df_tagcnt_std[col][idx] != 0]))
            matrix_H.append(list_H)
            list_H =[]
        else:
            list_H.append(sum([-df_tagcnt_std[col][idx] * math.log(df_tagcnt_std[col][idx], 2) for idx in range(len(df_tagcnt_std[col])) if df_tagcnt_std[col][idx] != 0]))


    df_H = pd.DataFrame(matrix_H, columns=[i for i in range(1, num_file+1)], index=languages).T
    df_H.loc['avr'] = [df_H[col].mean() for col in df_H.columns]
    df_H.loc['std'] = [df_H[col].std() for col in df_H.columns]
    
    return df_H
