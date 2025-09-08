from database import Database
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import sys

def get_chunks_from_file(path, chunk_size, overlap):
    print(f"Reading file {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    i = 0
    while i*chunk_size-i*overlap < len(content):
        chunk = content[i*chunk_size-i*overlap:(i+1)*chunk_size-i*overlap]
        chunks.append(chunk)
        i += 1
    
    return chunks
    

def get_embeddings_for_chunks(chunks):
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    client = genai.Client(api_key=API_KEY)

    embeddings = []
    num_chunks = len(chunks)
    chunks_per_request = 6
    num_iterations = (num_chunks+chunks_per_request-1) // chunks_per_request
    
    for i in range(num_iterations):
        chunks_group = chunks[i:i+chunks_per_request]

        print("Sending request to embedding model...")
        result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunks_group,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY", output_dimensionality=768)
        )
        group_embeddings = result.embeddings
        for emb in group_embeddings:
            embeddings.append(emb.values)
        print("Request processed...")
    
    return embeddings


def main():
    file_path = sys.argv[1]
    # chunk_size = int(sys.argv[2])

    chunk_size = 1000
    overlap = 0
    # overlap = int(sys.argv[3])

    print(f"File to embed: {file_path}")

    db = Database()
    db.create_vector_table()

    print("Spliting into chunks...")
    chunks = get_chunks_from_file(file_path, chunk_size, overlap)

    print("Calculating embeddings...")
    embeddings = get_embeddings_for_chunks(chunks)

    print("Writing to database...")
    for chu, emb in zip(chunks, embeddings):
        db.write_vector(chu, emb)


if __name__ == "__main__":
        main()