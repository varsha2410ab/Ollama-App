# 🍔 Food Facts RAG Demo

A cute, foodie-themed Retrieval-Augmented Generation (RAG) web application built using **Streamlit** and **Ollama**.

## What is this project?
This application allows users to ask questions about food. Behind the scenes, it uses a local vector database populated with fun food facts. When a question is asked, the app searches the database for the most relevant facts and provides them to a local AI model (Llama-3.2) as context, ensuring the AI gives accurate and relevant answers!

## Features
- **Local AI Processing**: Fully private and runs on your own hardware using Ollama.
- **Vector Search**: Embeds chunks of text and uses cosine similarity to find the most relevant facts to your query.
- **Cute Foodie UI**: Custom CSS styling with soft pink and yellow colors, rounded edges, and culinary emojis.
- **Streaming Responses**: Watch the AI type out its culinary wisdom in real-time.

## Requirements
- Python
- [Ollama](https://ollama.com) installed and running locally.
- Required Ollama models:
  - `ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` (for embeddings)
  - `ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` (for generation)

## How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit-app.py
```
