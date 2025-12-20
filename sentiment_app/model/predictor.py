from tokenizers import Tokenizer
import onnxruntime as ort
from src.scripts.settings import Settings
import numpy as np
from pathlib import Path

settings = Settings()

def load_tokenizer() -> Tokenizer:
    path = Path(f"{settings.tokenizer_path}/tokenizer.json")
    if not path.exists():
        raise FileNotFoundError(f"Embedding model not found at: {path}")
    
    tokenizer = Tokenizer.from_file(str(path))
    return tokenizer

def load_embedding_model() -> ort.InferenceSession:

    path = Path(settings.onnx_embedding_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding model not found at: {path}")
    
    ort_session = ort.InferenceSession(path)
    return ort_session


def load_classifier() -> ort.InferenceSession:
    path = Path(settings.onnx_classifier_path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier not found at: {path}")
    
    ort_session = ort.InferenceSession(path)
    return ort_session

SENTIMENT_MAP = {
    0: "negative", 
    1: "neutral", 
    2: "positive"
}

class PredictionModel:

    def __init__(self):
        self.tokenizer = load_tokenizer()
        self.embedding_model = load_embedding_model()
        self.classifier = load_classifier()

    def predict(self, sentence: str) -> str:
        encoded = self.tokenizer.encode(sentence)

        # prepare numpy arrays for ONNX
        input_ids = np.array([encoded.ids])
        attention_mask = np.array([encoded.attention_mask])

        # run embedding inference
        embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        embeddings = self.embedding_model.run(None, embedding_inputs)[0]

        # run classifier inference
        classifier_input_name = self.classifier.get_inputs()[0].name
        classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
        prediction = self.classifier.run(None, classifier_inputs)[0]

        label = SENTIMENT_MAP.get(prediction[0], "unknown") # return this label as response
        return label