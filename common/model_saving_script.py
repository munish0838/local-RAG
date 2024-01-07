from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()
embedder_locator: str = os.environ.get("EMBEDDER_LOCATOR", "intfloat/e5-large-v2"),
model = SentenceTransformer('all-mpnet-base-v2')
model.save(embedder_locator)
