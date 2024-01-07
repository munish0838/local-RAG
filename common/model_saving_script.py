from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()
model = SentenceTransformer('all-mpnet-base-v2')
model.save("/mnt/c/Users/Munish/Desktop/pathway/mpnet")
