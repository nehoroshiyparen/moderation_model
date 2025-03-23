import pandas as pd
import kagglehub

path = kagglehub.dataset_download("../raw/jigsaw-toxic-comment-classification-challenge")
print("Path to jigsaw-dataset files:", path)

jigsaw_df = pd.read_csv(path + '/train.csv')

jigsaw_df = jigsaw_df[['comment_text', 'toxic']]  
jigsaw_df = jigsaw_df.rename(columns={'comment_text': 'text'})  
jigsaw_df['language'] = 'en'

rutoxic_df = pd.read_csv('../raw/RuToxic/dataset.csv')  

rutoxic_df = rutoxic_df[['comment', 'toxic']]  
rutoxic_df = rutoxic_df.rename(columns={'comment': 'text'})  

rutoxic_df['language'] = 'ru'

jigsaw_df.to_csv('../processed/jigsaw_cleaned.csv', index=False)
rutoxic_df.to_csv('../processed/rutoxic_cleaned.csv', index=False)

print("Prepared Jigsaw and RuToxic datasets saved.")