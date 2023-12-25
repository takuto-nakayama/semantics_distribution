import pandas as pd
import itertools
import math

def js_div(df_tagcnt:pd.DataFrame):
    df_tagcnt_std = df_tagcnt.apply(lambda x: x/x.sum()) # 
    df_JS = pd.DataFrame(0, index=df_tagcnt.columns, columns=df_tagcnt.columns) # for recording
    combo = itertools.combinations(df_tagcnt.columns, 2) # combination of the files

    for cmb in combo:
        cnt1 = df_tagcnt[cmb[0]].sort_values(ascending=False)
        cnt2 = df_tagcnt[cmb[1]].sort_values(ascending=False)

        list_r = [(p + q) / 2 for (p,q) in zip(cnt1, cnt2)]
        KL_pr = sum([p * math.log(p/r, 2) for (p,r) in zip(cnt1, list_r) if p != 0])
        KL_qr = sum([q * math.log(q/r, 2) for (q,r) in zip(cnt2, list_r) if q != 0])
        JS_div = (KL_pr + KL_qr) / 2

        df_JS[cmb[0]][cmb[1]] = JS_div
        df_JS[cmb[1]][cmb[0]] = JS_div

    return df_JS