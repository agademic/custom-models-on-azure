import os
import logging
import json
import uuid

from flair.models import MultitaskModel

from flair.splitter import SpacySentenceSplitter, SegtokSentenceSplitter
from flair.tokenization import SpacyTokenizer

from pydantic import BaseModel
from typing import List

# from azureml.core import Model


class PredictionRecord(BaseModel):
    record_id: str
    raw: str
    tag: str
    score: float
    start: int
    end: int
    context: str
    relations_from: List[str]
    relations_to: List[str]

class ExtractionModelResponse3(BaseModel):
    processId: str
    predictions: List[PredictionRecord]


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "multitask_model_small.pt"
    )
    # deserialize the model file back into a flair model
    model = MultitaskModel.load(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received: ", raw_data)
    process_id = json.loads(raw_data)["processId"]
    text = json.loads(raw_data)["text"]

    splitter = SegtokSentenceSplitter()#SpacySentenceSplitter(model='xx_sent_ud_sm', tokenizer=SpacyTokenizer(model='xx_sent_ud_sm'))
    doc = splitter.split(text)

    model.predict(doc, verbose=True, mini_batch_size=8)

    entities = []
    span_id_map = {}
    for sentence in doc:
        for span in sentence.get_spans('ner'):
            span_id = str(uuid.uuid4())
            span_id_map[span] = span_id
            span.id = span_id  # assign an unique ID to span
            entities.append(span)

    relations = []
    for sentence in doc:
        relations.extend(sentence.get_relations('relation'))

    predictions = []
    for span in entities:
        rel_from_ids = [span_id_map[rel.first] for rel in relations if rel.second == span]
        rel_to_ids = [span_id_map[rel.second] for rel in relations if rel.first == span]

        prediction_record = PredictionRecord(
            record_id=span.id,
            raw=span.text,
            tag=span.get_label('ner').value,
            score=span.score,
            start=span.start_position,
            end=span.end_position,
            context=span.sentence.to_original_text(),
            relations_from=rel_from_ids,
            relations_to=rel_to_ids
        )
        predictions.append(prediction_record)
    
    response = ExtractionModelResponse3(processId=process_id, predictions=predictions)
    logging.info("Request processed: ", response)
    return response.json()