import pdfplumber
from PIL import Image
import io
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PayloadSchemaType

# Initialize Qdrant client
qdrant = QdrantClient("localhost", port=6333)

# Initialize LLaMA model and tokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained('llama-3b')
llama_model = LlamaForCausalLM.from_pretrained('llama-3b')

# Sentence transformer for embeddings (use an appropriate model)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create a collection in Qdrant for RAG
qdrant.create_collection(
    collection_name="pdf_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Function to extract text and images from a PDF
def extract_pdf_content(pdf_path):
    texts = []
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            if text:
                texts.append(text)

            # Extract images
            for image in page.images:
                x0, y0, x1, y1 = image["x0"], image["y0"], image["x1"], image["y1"]
                img = page.within_bbox((x0, y0, x1, y1)).to_image()
                img_data = io.BytesIO()
                img.save(img_data, format="PNG")
                img_data.seek(0)
                pil_image = Image.open(img_data)
                images.append(pil_image)

    return texts, images

# Function to index text into Qdrant
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

# Function to retrieve relevant documents from Qdrant
def retrieve_relevant_text(query):
    query_embedding = embedder.encode(query).tolist()
    results = qdrant.search(
        collection_name="pdf_collection",
        query_vector=query_embedding,
        limit=3  # retrieve top 3 results
    )
    return [result.payload['text'] for result in results]

# Function to generate response using LLaMA model
def generate_answer(context, query):
    input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    input_ids = llama_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = llama_model.generate(input_ids, max_length=500)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function to run RAG on a PDF
def rag_system_on_pdf(pdf_path, query):
    texts, images = extract_pdf_content(pdf_path)

    # Index extracted texts
    index_texts(texts)

    # Retrieve relevant texts based on the query
    relevant_texts = retrieve_relevant_text(query)

    # Combine relevant texts to form a context for generation
    context = " ".join(relevant_texts)

    # Generate a response using LLaMA
    answer = generate_answer(context, query)
    
    return answer, images  # return the answer and extracted images

# Example usage
pdf_path = "example.pdf"
query = "What is the main topic discussed in the document?"
answer, images = rag_system_on_pdf(pdf_path, query)

print("Answer:", answer)
for img in images:
    img.show()  # Display the extracted images
