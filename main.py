import streamlit as st
import pdfplumber
from PIL import Image
import io
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaImageProcessor
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Initialize models
model_name = "meta-llama/Llama-3.2-11b-chat-hf"
llama_tokenizer = LlamaTokenizer.from_pretrained('model_name')
llama_model = LlamaForCausalLM.from_pretrained('model_name', device_map="auto", torch_dtype=torch.float16)

llama_image_processor=LlamaImageProcessor.from_pretrained(model_name)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient("localhost", port=6333)

# Streamlit UI components
st.title("LLaMA 3.2 11B Multimodal RAG System")
st.write("Upload a PDF file and ask questions about its contents.")

# File upload via Streamlit
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
query = st.text_input("Ask your own query:")

# Create a Qdrant collection for indexing
def create_qdrant_collection():
    qdrant.create_collection(
        collection_name="pdf_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Convert PDF to images
def convert_pdf_to_images(pdf_path):
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            img = page.to_image(resolution=300).original
            img_data = io.BytesIO()
            img.save(img_data, format="PNG")
            img_data.seek(0)
            pil_image = Image.open(img_data)
            images.append(pil_image)
    return images

# Use LLaMA 3.2 11B to convert images to text
def convert_images_to_text(images):
    image_texts = []
    for image in images:
        inputs = llama_image_processor(images=image, return_tensors="pt").to(llama_model.device)
        outputs = llama_model.generate(**inputs, max_new_tokens=100)
        text_description = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        image_texts.append(text_description)
    return image_texts

# Index extracted text into Qdrant
def index_texts(texts):
    for i, text in enumerate(texts):
        embedding = embedder.encode(text).tolist()
        payload = {"page_num": i, "text": text}
        qdrant.upsert(
            collection_name="pdf_collection",
            points=[{
                "id": i,
                "vector": embedding,
                "payload": payload
            }]
        )

# Retrieve relevant text from Qdrant
def retrieve_relevant_text(query):
    query_embedding = embedder.encode(query).tolist()
    results = qdrant.search(
        collection_name="pdf_collection",
        query_vector=query_embedding,
        limit=3
    )
    return [result.payload['text'] for result in results]

# Generate an answer using LLaMA 3.2 11B
def generate_answer(context, query):
    input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = llama_tokenizer(input_text, return_tensors="pt").to(llama_model.device)
    outputs = llama_model.generate(**inputs, max_new_tokens=500)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main RAG pipeline
def rag_system_on_pdf(pdf_path, query):
    images = convert_pdf_to_images(pdf_path)
    texts_from_images = convert_images_to_text(images)
    index_texts(texts_from_images)
    relevant_texts = retrieve_relevant_text(query)
    context = " ".join(relevant_texts)
    answer = generate_answer(context, query)
    return answer, images

# Execute when file and query are provided
if uploaded_file is not None and query:
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    answer, images = rag_system_on_pdf(pdf_path, query)
    
    st.write(f"**Answer to your query:** {answer}")
    
    st.write("**Extracted Images:**")
    for img in images:
        st.image(img)

    st.write("**Extracted and indexed text from images:**")
    for i, text in enumerate(convert_images_to_text(images)):
        st.write(f"Image {i+1}: {text}")

