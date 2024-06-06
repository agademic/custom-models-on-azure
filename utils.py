import json
import requests
from ast import literal_eval
import numpy as np


def cosine_similarity(a, b):
    if type(a) is not np.ndarray:
        a = np.array(a)
        b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_text(text: str, endpoint_uri: str):
    request_data = json.dumps({'data': text})
    response = requests.post(endpoint_uri, data=request_data, headers={'Content-Type': 'application/json'})
    response = literal_eval(response.json())
    return response["predictions"]