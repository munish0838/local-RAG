import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_host = os.environ.get("HOST", "0.0.0.0")
api_port = int(os.environ.get("PORT", 8080))

# Streamlit UI elements
st.title("Local-RAG")

question = st.text_input(
    "Search for something",
    placeholder="What data are looking for?"
)


if question:
    print("Query received")
    url = f'http://{api_host}:{api_port}/'
    data = {"query": question}
    print("Sending for response")
    response = requests.post(url, json=data)
    
    print("Waiting for response")
    if response.status_code == 200:
        st.write("### Answer")
        st.write(response.json())
        print(f"Printing the response: {response}")
    else:
        st.error(f"Failed to send data to Pathway API. Status code: {response.status_code}")
