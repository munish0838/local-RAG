from dotenv import load_dotenv
import os
from llm_app.model_wrappers import OpenAIEmbeddingModel, OpenAIChatGPTModel
from llm_app.model_wrappers import HFTextGenerationTask, SentenceTransformerTask
load_dotenv()


embedder_locator = os.environ.get("EMBEDDER_LOCATOR", "text-embedding-ada-002")
api_key = os.environ.get("HF_API_TOKEN", "")
model_locator = os.environ.get("MODEL_LOCATOR", "google/flan-t5-small")
max_tokens = int(os.environ.get("MAX_TOKENS", 200))
temperature = float(os.environ.get("TEMPERATURE", 0.0))


def openai_embedder(data):
    embedder = OpenAIEmbeddingModel(api_key=api_key)
    return embedder.apply(text=data, locator=embedder_locator)


def openai_chat_completion(prompt):
    # model = OpenAIChatGPTModel(api_key=api_key)
    model = HFTextGenerationTask(model=model_locator, device="cpu")
    return model.apply(
            prompt,
            # locator=model_locator,
            # temperature=temperature,
            max_new_tokens=max_tokens,
        )
