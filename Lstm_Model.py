import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Load the model
model = load_model('sentiment_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

# Limit the tokenizer to the same vocabulary size as the embedding layer
tok.num_words = 500

# Load the max_len
with open('max_len.txt', 'r') as f:
    max_len = int(f.read())



def predict(text, include_neutral=True):
    # Handle case when `text` is a single string
    if isinstance(text, str):
        text = [text]

    # Tokenize text
    sequences = tok.texts_to_sequences(text)
    
    # Replace out-of-vocabulary tokens (e.g., any token >= 500) with 0 or an OOV index
    sequences = [[min(token, 499) for token in seq] for seq in sequences]

    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
    
    # Predict using the model
    score = model.predict(sequences_matrix)[0][0]
    
    # Determine sentiment label
    if score >= 0.5:
        label = "Positive"
    else:
        label = "Negative"

    return {"label": label, "score": float(score)}

