import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from huggingface_hub import notebook_login

# --- Configuration ---
NEWSPAPERS = ['lf_all', 'aal_all', 'od_all_clean', 'thi_all', 'vib_all']
EXTRA_NEWSPAPERS = ['aar_ext', 'rib_all', 'sla_all']
PATH_ROOT = '../../../ROOT/DATA/NEWSPAPERS/article_embs/embeddings_e5/'
GOLD_PATH = '../results/predicted_sample_tfidf_embeddings(1).csv'
HF_DATASET = "chcaa/periphery-aviser-e5"

# --- Load and Combine Base Newspaper Embeddings ---
def load_and_concat_embeddings(newspaper_list):
    dfs = []
    for paper in newspaper_list:
        print(f"Loading {paper}...")
        ds = Dataset.load_from_disk(f'{PATH_ROOT}{paper}')
        df = ds.to_pandas()
        dfs.append(df)
    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined

print("Loading main newspaper datasets...")
final_df = load_and_concat_embeddings(NEWSPAPERS)

print("Filtering invalid embeddings...")
final_df['embedding_shape'] = final_df['embedding'].apply(lambda x: np.array(x).shape)
expected_dim = final_df['embedding_shape'].max()[0]
final_df = final_df[final_df['embedding'].apply(lambda x: np.array(x).shape == (expected_dim,))].copy()

final_df['newspaper'] = final_df['article_id'].str.extract(r'^(.*?)_')

print("Loading and applying gold standard labels...")
gold_set = pd.read_csv(GOLD_PATH, index_col=0, sep=';')
gold_set = gold_set.applymap(lambda x: x.strip() if isinstance(x, str) else x)

gold_set["true_label"] = gold_set.apply(
    lambda row: row["predicted_category_tf_idf"] if row["evaluation_tf_idf"] == "t" else
                (row["predicted_category_embs"] if row["evaluation_embs"] == "t" else row["true_label"]),
    axis=1
)

final_df = final_df.merge(gold_set[['article_id', 'true_label']], on='article_id', how='left')
final_df['clean_category'] = final_df['true_label'].fillna(final_df['clean_category'])
final_df.drop(columns=['true_label'], inplace=True)
final_df['label_type'] = final_df['clean_category'].apply(lambda x: 'predicted' if x == -1 else 'gold')

print("Balancing training dataset...")
training_df = final_df[final_df['label_type'] == 'gold']
lol_df = training_df[training_df['newspaper'] == 'lol']

lol_balanced = lol_df.groupby('clean_category', group_keys=False).apply(
    lambda x: x.sample(n=400, replace=True) if len(x) >= 400 else x
).reset_index(drop=True)

other_df = training_df[training_df['newspaper'] != 'lol']
filtered_df = pd.concat([lol_balanced, other_df], ignore_index=True)

print("Splitting into train/test sets...")
train_df, test_df = train_test_split(
    filtered_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=filtered_df['clean_category']
)

X_train = np.vstack(train_df['embedding'].values)
y_train = train_df['clean_category'].values
X_test = np.vstack(test_df['embedding'].values)
y_test = test_df['clean_category'].values

print("Training logistic regression classifier on embeddings...")
clf_embs = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
clf_embs.fit(X_train, y_train)

print("Loading extra newspaper datasets for prediction...")
aar_df = load_and_concat_embeddings(EXTRA_NEWSPAPERS)

print("Filtering invalid embeddings for prediction set...")
aar_df['embedding_shape'] = aar_df['embedding'].apply(lambda x: np.array(x).shape)
expected_dim = aar_df['embedding_shape'].max()[0]
aar_df = aar_df[aar_df['embedding'].apply(lambda x: np.array(x).shape == (expected_dim,))].copy()

aar_df['newspaper'] = aar_df['article_id'].str.extract(r'^(.*?)_')

if 'clean_category_y' in aar_df.columns and 'clean_category_x' in aar_df.columns:
    aar_df = aar_df.drop(columns=['clean_category_y']).rename(columns={'clean_category_x': 'clean_category'})

print("Predicting labels for new articles...")
aar_df['label_type'] = aar_df['clean_category'].apply(lambda x: 'predicted' if x == -1 else 'gold')
pred_aar_df = aar_df[aar_df['label_type'] == 'predicted']

X_pred_embs = np.vstack(pred_aar_df['embedding'].values)
pred_aar_df['predicted_category_embs'] = clf_embs.predict(X_pred_embs)

aar_df = aar_df.merge(pred_aar_df[['article_id', 'predicted_category_embs']], on='article_id', how='left')
aar_df['clean_category'] = aar_df['predicted_category_embs'].fillna(aar_df['clean_category'])
aar_df.drop(columns=['predicted_category_embs'], inplace=True)

dataset_extra = Dataset.from_pandas(aar_df)
dataset_extra.save_to_disk('../../../ROOT/DATA/NEWSPAPERS/backup_extra_newspapers_250624')
print("Saved dataset to disk.")

print("Loading existing HF dataset and merging with new predictions...")
existing_dataset = load_dataset(HF_DATASET, split='train')
existing_df = existing_dataset.to_pandas().drop(columns=['nøgle', 'category'])

merged_df = pd.concat([existing_df, aar_df], ignore_index=True)
dataset_to_push = Dataset.from_pandas(merged_df, preserve_index=False)

print("Pushing updated dataset to Hugging Face Hub...")
dataset_to_push.push_to_hub(HF_DATASET, private=True)
print("Upload complete.")