import os
from dotenv import load_dotenv
from pathway.stdlib.ml.index import KNNIndex
from common.openaiapi_helper import openai_embedder
from llm_app.model_wrappers import HFTextGenerationTask, SentenceTransformerTask
load_dotenv()
embedder_locator: str = os.environ.get("EMBEDDER_LOCATOR", "intfloat/e5-large-v2"),
# embedding_dimension = int(os.environ.get("EMBEDDING_DIMENSION", 1536))
embedder = SentenceTransformerTask(model=embedder_locator, device="cpu")
embedding_dimension = len(embedder(""))

def embeddings(context, data_to_embed):
    # Enriched documents here => Convert to vector
    return context + context.select(vector=embedder(data_to_embed))



def index_embeddings(embedded_data):
    return KNNIndex(embedded_data.vector, embedded_data, n_dimensions=embedding_dimension)
