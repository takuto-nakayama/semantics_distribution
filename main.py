languages = ['Chinese',
             'Dutch',
             'English',
             'Finish',
             'French',
             'Italian',
             'Portuguese',
             'Spanish'
             ]
num_file = 10
file_id = 'sample'


import tag_counter as tc
import js_divergence as js
import entropy as ent
import visualize as vis


df_tagcnt = tc.tag_cnt(languages, num_file, Z=0)
df_tagcnt.to_csv(f'result/{file_id}_tagcount.csv')

df_JSdiv = js.js_div(df_tagcnt)
df_JSdiv.to_csv(f'result/{file_id}_JSdiv.csv')

df_entropy = ent.entropy(df_tagcnt, languages, num_file)
df_entropy.to_csv(f'result/{file_id}_entropy.csv')

mds = vis.mds(df_JSdiv, languages, num_file)
df_mds = mds[1]
df_mds.to_csv(f'result/{file_id}_mds.csv')
fig_mds = mds[0]
fig_mds.write_html(f'result/{file_id}_mds.html')

tsne = vis.tsne(df_tagcnt, languages, num_file, perplexity=5)
df_tsne = tsne[1]
df_tsne.to_csv(f'result/{file_id}_tsne.csv')
fig_tsne = tsne[0]
fig_tsne.write_html(f'result/{file_id}_tsne.html')

fig_mds.show()
fig_tsne.show()