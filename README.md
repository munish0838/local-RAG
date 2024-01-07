# Local-RAG

It allows you to answer questions based on your test using SOTA model, locally on your system, without worrying about OpenAI API costs or data privacy

## Demo

See how the tool works:
[Video Link](https://drive.google.com/file/d/1LQlQ1oKDfjQB6L3Xb--LqKIw9DBHvtwa/view?usp=sharing)

As you can see the LLM App enables AI-powered search from multiple unstructured documents like tax information from different countries, and indexes input data in real-time just after you upload files to the storage.

## How to run the tool

1. Create `.env` file in the root directory of the project and add following environemnt variables
2. Clone this repo 

```bash
EMBEDDER_LOCATOR={Path to save local sentence transformer embedding model}
EMBEDDING_DIMENSION=1024
MAX_TOKENS=200
TEMPERATURE=0.0
PATH_TO_DROPBOX={Path to local document folder}
MODEL_LOCATOR={PAth to model on huggingface}

```
3. Install requirements by using `pip install -r requirements.txt`
4. Run `model_saving_script.py` to save the model in required path
4. In `embedder.py` file, specify the sentence tranformer model location on local 
4. From the project root folder, open your terminal and run `python main.py`.
5. After API runs succesfully in terminal, now run `streamlit run ui.py`.
6. Navigate to `localhost:8501` on your browser.