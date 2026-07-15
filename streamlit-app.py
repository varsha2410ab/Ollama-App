import streamlit as st
import ollama
import os

st.set_page_config(page_title="Food Facts RAG", layout="centered", page_icon="🍔")

# Inject Custom CSS for the Food Theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #fff0f5;
        background-image: linear-gradient(135deg, #fff0f5 0%, #ffe4e1 100%);
        font-family: 'Nunito', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Headers */
    h1 {
        color: #ff6b81 !important;
        text-shadow: 2px 2px 0px #ffeaa7;
        font-weight: 800 !important;
    }
    h2, h3, h4 {
        color: #ff4757 !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fff9c4 !important; /* cute butter yellow */
        border-right: 3px dashed #ff9ff3;
    }
    /* Chat input container */
    [data-testid="stChatInput"] {
        border-radius: 30px !important;
        border: 2px solid #ff9ff3 !important;
        box-shadow: 0 4px 6px rgba(255, 159, 243, 0.2) !important;
    }
    /* Success / Info Alerts */
    [data-testid="stAlert"] {
        border-radius: 15px;
        background-color: #ffffff;
        border: 2px solid #ff6b81;
        box-shadow: 3px 3px 0px #ff6b81;
        color: #2f3542;
    }
    /* Body text */
    p {
        color: #2f3542;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍔 Food Facts RAG Demo")
st.write("Welcome to the **Food Facts** demo! Ask a question about food, and we'll retrieve relevant chunks of culinary knowledge to answer it for you! 🍕🌮🥗")

# Model Configurations
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# --- 1. Vector DB Setup & Caching ---
@st.cache_resource
def initialize_vector_db():
    """Loads the dataset and computes embeddings once, caching the results."""
    dataset_path = 'food-facts.txt'
    
    # Check if file exists to prevent crashing
    if not os.path.exists(dataset_path):
        # Creating a fallback file for demonstration if it doesn't exist
        with open(dataset_path, 'w') as f:
            f.write("Apples float in water because they are 25% air.\n")
            f.write("Bananas are berries, but strawberries aren't.\n")
            f.write("Honey never spoils.\n")
            f.write("Carrots were originally purple, not orange.\n")
            f.write("Peanuts aren't technically nuts; they are legumes.\n")

    with open(dataset_path, 'r',encoding="utf-8") as file:
        dataset = file.readlines()
    
    vector_db = []
    
    # Progress bar for visual feedback during startup embedding generation
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(dataset):
        chunk = chunk.strip()
        if not chunk:
            continue
        status_text.text(f"Embedding chunk {i+1}/{len(dataset)}...")
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            vector_db.append((chunk, embedding))
        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}")
            return []
        progress_bar.progress((i + 1) / len(dataset))
        
    # Clear the loading indicators when done
    status_text.empty()
    progress_bar.empty()
    return vector_db

# Initialize the database
with st.spinner("Initializing Vector Database and generating embeddings..."):
    VECTOR_DB = initialize_vector_db()


# --- 2. Helper Functions ---
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if int(norm_a * norm_b) == 0:
        return 0
    return dot_product / (norm_a * norm_b)

## Retrieval function that takes a user query, computes its embedding, and finds the most similar chunks in the VECTOR_DB based on cosine similarity.
def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]


# --- 3. UI and Interaction ---

# Sidebar to show the retrieved context chunks behind the scenes
with st.sidebar:
    st.header("Database Info")
    st.success(f"Loaded {len(VECTOR_DB)} items from dataset.")
    st.markdown("---")
    st.subheader("Retrieved Context Window")
    context_placeholder = st.empty()
    context_placeholder.info("Ask a question to see the matching context chunks here!")


# User Input
input_query = st.chat_input("Ask me a question about food... e.g., Do apples float?")

if input_query:
    # Perform Retrieval
    retrieved_knowledge = retrieve(input_query)
    
    # Update the sidebar dynamically to display retrieved chunks and their scores
    with context_placeholder.container():
        for chunk, similarity in retrieved_knowledge:
            st.markdown(f"**Score:** `{similarity:.2f}`\n\n*{chunk}*")
            st.markdown("---")

    # Construct the instruction prompt
    instruction_prompt = f"""You are a helpful and enthusiastic culinary expert.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
"""

    st.subheader(f"🧑‍🍳 Response to: *{input_query}*")
    
    # Dynamic placeholder for streaming the response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        with st.spinner("Cooking up an answer..."):
            stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': input_query},
                ],
                stream=True,
            )
            
            # Iterate over stream chunks and dynamically update the UI
            for chunk in stream:
                token = chunk['message']['content']
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
        # Final update to remove the cursor icon
        response_placeholder.markdown(full_response)
        
    except Exception as e:
        st.error(f"Error generating chat response: {e}")