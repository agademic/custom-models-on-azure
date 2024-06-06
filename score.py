import json
from sentence_transformers import SentenceTransformer
from azureml.core import Model


def init():
    global model, tokenizer
    # The model and tokenizer are loaded from a path where the model is registered in Azure
    model_path = Model.get_model_path('bge-m3')
    model = SentenceTransformer(model_path)


def run(raw_data):
    try:
        # Convert the JSON data into a Python dictionary
        data = json.loads(raw_data)
        texts = data['data']

        # Prediction
        predictions = model.encode(texts)
        predictions = predictions.tolist() # convert to a list to be able to serialize it

        # Return predictions in a JSON format
        return json.dumps({"predictions": predictions})

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
