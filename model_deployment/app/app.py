import string 
import os 
import gensim
from gensim import corpora 
from gensim.models import LdaModel 
from gensim.models.phrases import Phraser
import joblib 
import numpy as np
from fastapi import FastAPI, HTTPException
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from pydantic import BaseModel 
import tensorflow as tf 
from tensorflow.keras.preprocessing.sequence import pad_sequences 

nltk.download("stopwords", quite=True)
nltk.download("wordnet", quite=True)
nltk.download("omw-1.4", quite=True)

app = FastAPI(title= "Women's Clothing sentiment analysis API", version="1.0", 
              description="Sentiment analysis and topic modelling")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

try:
   model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
   tokenizer = joblib.load(os.path.join(MODEL_DIR, 'tokenizer.joblib'))
   lb = joblib.load(os.path.join(MODEL_DIR, "lb.joblib"))
except Exception as e:
    model = None
    tokenizer = None 
    lb = None
    print(f'Could not load LSTM model/toenizer artifacts: {e}')

try:
    lda_model = LdaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))
    dictionary = corpora.Dictionary.load(os.path.join(MODEL_DIR, "gensim_dictionary.dict"))
    bigram_model = Phraser.load(os.path.join(MODEL_DIR, "bigram_model.pkl"))
except Exception:
    lda_model = None
    dictionary = None 
    bigram_model = None

def preprocess_text(text: str) -> str:
    if not text or not isinstance(text,str):
        return " "
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    stop_words = set(stopwords.words("english"))
    cleaned_text = [w for w in text.split() if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    corpus = [lemmatizer.lemmatize(w) for w in cleaned_text]
    return " ".join(corpus)

def extract_lda_topics(text:str):
    if not lda_model or not dictionary or not bigram_model:
        return []
    
    tokens = [word for word in text.lower().split() if word.isalpha() and len(word) > 2]
    bigrams_tokens = bigram_model[tokens]
    custom_words = [
      "dress",
      "would",
      "top",
      "shirt",
      "look",
      "skirt",
      "really",
      "wear",
      "back",
      "item",
      "like",
      "fit",
      "size",
  ]
    filtered_tokens = [w for w in bigrams_tokens if w not in custom_words]
    bow = dictionary.doc2bow(filtered_tokens)
    topic_distribution = lda_model.get_document_topics(bow)
    formatted_topics= []
    for topic_id, prob in topic_distribution:
        topic_words = lda_model.show_topic(topic_id, topn=3)
        formatted_topics.append({
            "topic_id": int(topic_id),
            "probability": float(prob),
            "top_words": [word for word, _ in topic_words]
        })
    return formatted_topics

 
class ReviewRequest(BaseModel):
    text: str 

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "lstm_model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "label_binarizer_loaded": lb is not None,
        "lda_model_loaded": lda_model is not None
    }

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    if not model or not tokenizer or not lb:
        raise HTTPException(
            status_code=500, detail = "Model or preprocessing artifacts not loaded."
        )
    processed_review = preprocess_text(request.text)
    sequences = tokenizer.texts_to_sequences([processed_review])
    padded = pad_sequences(sequences, maxlen=63, padding='pre', truncating="post")

    prediction = model.predict(padded)
    label_map = lb.classes_
    predicted_idx = int(np.argmax(prediction))
    sentiment = str(label_map[predicted_idx])
    confidence = float(np.max(prediction) * 100)

    topics = []
    if sentiment.lower() in ["bad", "negative", "positive"]:
        topics =  extract_lda_topics(request.text)


    return {
        "orginal_review": request.text,
        "processed_review": processed_review,
        "sentiment": sentiment,
        "confidence_percent": round(confidence, 2),
        "review_topics": topics
    }