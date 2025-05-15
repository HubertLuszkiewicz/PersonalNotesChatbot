from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

class ChatbotClient:
    def __init__(self, greeting="Hello, how can I help you today?"):
        # Load API KEY
        load_dotenv()
        API_KEY = os.getenv("API_KEY")
        self.greeting = greeting
        self.chunks = []
        self.embeddings_for_chunks = []
        self.chunks_similarity = []
        self.client = genai.Client(api_key=API_KEY)

        # Read file and split it into chunks
        with open("rainforest.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        chunk_size = 200
        overlap = 30
        i = 0
        while i*chunk_size-i*overlap < len(content):
            chunk = content[i*chunk_size-i*overlap:(i+1)*chunk_size-i*overlap]
            self.chunks.append(chunk)
            i += 1

        # Generate embeddings for chunks
        for chunk in self.chunks:
            result = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=[chunk],
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            [embedding] = result.embeddings
            self.embeddings_for_chunks.append(embedding)
    
    def get_response(self, user_question):
        # Embedd user question
        result = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=[user_question],
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
        [user_question_embedding] = result.embeddings

        # Compare embeddings for chunks with user question embedding
        for i, emb in enumerate(self.embeddings_for_chunks):
            similarity = cosine_similarity([emb.values], [user_question_embedding.values])
            self.chunks_similarity.append((self.chunks[i], similarity[0][0]))

        # Choose best 20% of chunks
        self.chunks_similarity.sort(key=lambda x: x[1], reverse=True)
        best_chunks = self.chunks_similarity[:int(len(self.chunks_similarity)*0.2)+1]

        # Create context by combining chosen chunks
        context = ''.join([str(chu[0]) for chu in best_chunks])

        # Combine user question and matched chunks to create prompt for Gemini LLM
        prompt = f"""
            {context}
            {user_question}
        """

        # Create system prompt
        system_prompt = """
            You are an AI assistant. Answer the user's question based ONLY on the 
            provided context documents. If the answer is not found in the context, 
            say "I cannot answer this question based on the provided documents." 
            Do not use any prior knowledge or information outside of the context.
        """

        # Get response from model
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt),
            contents=prompt
        )

        return f"{response.text.strip()}\n\Chunks used:\n{context}"

    def get_greeting(self):
        return self.greeting

def main():
    chatbot = ChatbotClient()
    print("-" * 30)
    print(f"Model: {chatbot.get_greeting()}")
    print("-" * 30)

    while True:
        try:
            user_question = input("User: ")
        except EOFError:
            print("\nModel: Goodbye!")
            break

        if not user_question.strip():
            continue

        model_response = chatbot.get_response(user_question=user_question)
        print("-" * 30)
        print(f"Model: {model_response}", end="")
        print("-" * 30)

if __name__ == '__main__':
    main()