from sentence_transformers import SentenceTransformer, util
import pandas as pd, numpy as np
import pickle

books_cleaned = pd.read_csv("goodbooks-10k/books_cleaned.csv")
targets = books_cleaned.title
sen_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_1 = sen_model.encode(np.array(targets), convert_to_tensor=True)

#Store sentences & embeddings on disc
with open('model/embeddings.pkl', "wb") as fOut:
    pickle.dump(embedding_1, fOut, protocol=pickle.HIGHEST_PROTOCOL)