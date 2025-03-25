import pandas as pd
from datasets import Dataset

newspapers = ['aal_all', 'lf_all', 'od_all_clean', 'thi_all', 'vib_all']

path_root = '../../../DATA/NEWSPAPERS/'

for nsp in newspapers:
    # load embeddings
    embs = Dataset.load_from_disk(f'{path_root}embs/embeddings_e5/mean_output/{nsp}')
    embs = embs.to_pandas()
    # load articles and metadata
    texts = pd.read_csv(f'{path_root}ready_for_embs/{nsp}.csv', index_col=0)
    # merge embeddings with articles and metadata
    embs_texts = embs.merge(texts[['article_id', 'nøgle', 'text', 'category', 'article_length', 'characters']], on='article_id')
    #print(embs_texts.shape)
    dataset = Dataset.from_pandas(embs_texts)
    dataset.save_to_disk(f'{path_root}article_embs/embeddings_e5/{nsp}')