import streamlit as st
import pdfplumber
from PIL import Image
import io
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from llama_image_processor import LLAVAImageToText  # Assuming this is the LLAVA model

# Initialize models and clients
llama_tokenizer = LlamaTokenizer.from_pretrained('llama-3b')
llama_model = LlamaForCausalLM.from_pretrained('llama-3b')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
llava_image_processor = LLAVAImageToText()  # LLAVA model for converting images to text
qdrant = QdrantClient("localhost", port=6333)

# Streamlit UI components
st.title("LLaMA 3.2 Multimodal RAG System")
st.write("Upload a PDF file and ask questions about its contents.")

# File upload via Streamlit
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
query = st.text_input("Enter your query:")

# Create a Qdrant collection for indexing
def create_qdrant_collection():
    qdrant.create_collection(
        collection_name="pdf_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Function to convert PDF to images
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

# Convert images to text using LLAVA
def convert_images_to_text(images):
    image_texts = []
    for image in images:
        text_description = llava_image_processor.process(image)
        image_texts.append(text_description)
    return image_texts

# Index text into Qdrant
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

# Generate an answer using LLaMA
def generate_answer(context, query):
    input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    input_ids = llama_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = llama_model.generate(input_ids, max_length=500)
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
    # Convert uploaded file to path
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the PDF with RAG system
    answer, images = rag_system_on_pdf(pdf_path, query)
    
    # Display the results
    st.write(f"**Answer to your query:** {answer}")
    
    st.write("**Extracted Images:**")
    for img in images:
        st.image(img)

    st.write("**Extracted and indexed text from images:**")
    for i, text in enumerate(convert_images_to_text(images)):
        st.write(f"Image {i+1}: {text}")
