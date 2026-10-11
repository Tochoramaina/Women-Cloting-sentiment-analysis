import os 
import joblib 
import mlflow
import mlflow.tensorflow 
from gensim import corpora 
from gensim.models import ldaModel
from gensim.models.phrases import Phraser
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
mlflow.set_experiment("Womens_clothing_sentiment analysis and topic modelling")

with mlflow.start_run():
    mlflow.log_param("max_len", 63)
    mlflow.log_param("num_words", 10000)
    mlflow.log_param("optimizer", "adam")
    
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f'Directory "{MODEL_DIR}" not found')
    print("verifying and loading pre-trained model")
    

    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
        tokenizer = joblib.load(os.path.join(MODEL_DIR, "tokenizer.joblib"))
        lb = joblib.load(os.path.join(MODEL_DIR, "lb.joblib"))

        dictionary = corpora.Dictionary.load(os.path.join(MODEL_DIR, "gensim_dictionary.dict"))
        lda_model = ldaModel.load(os.path.join(MODEL_DIR, "lda_model.gensim"))
        bigram_model = Phraser.load(os.path.join(MODEL_DIR, "bigram_model.pkl"))
        print("All artifacts loaded successfully")
    except Exception as e:
        raise RuntimeError(f'Error loading model artifacts: {e}')

    mlflow.log_artifacts(MODEL_DIR, artifact_path = "model_artifacts")
    print("training complete! metrics, parameters and all artifacts logged")