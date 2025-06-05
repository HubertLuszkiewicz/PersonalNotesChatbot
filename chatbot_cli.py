from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from database import Database
import numpy as np

class ChatbotClient:
    def __init__(self, vector_db, greeting="Hello, how can I help you today?"):
        # Load API KEY
        load_dotenv()
        API_KEY = os.getenv("API_KEY")
        self.greeting = greeting
        self.chunks = []
        self.embeddings_for_chunks = []
        self.chunks_similarity = []
        self.client = genai.Client(api_key=API_KEY)
        self.vector_db = vector_db
    
    def get_response(self, user_question):
        # Embedd user question
        result = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=[user_question],
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
        [user_question_embedding] = result.embeddings

        # Get best chunks
        self.best_chunks = self.vector_db.get_best_chunks(np.array(user_question_embedding.values), 3)

        # Create context by combining chosen chunks
        context = ''.join(self.best_chunks)

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

        return response.text.strip()

    def get_greeting(self):
        return self.greeting


def main():
    db = Database()
    chatbot = ChatbotClient(vector_db=db)
    print("-" * 30)
    print(f"Chatbot: {chatbot.get_greeting()}")
    print("-" * 30)

    while True:
        try:
            user_question = input("User: ")
        except EOFError:
            print("\nChatbot: Goodbye!")
            db.close()
            break
        except KeyboardInterrupt:
            print("\nChatbot: Goodbye!")
            db.close()
            break

        if not user_question.strip():
            continue

        model_response = chatbot.get_response(user_question=user_question)
        print("-" * 30)
        print(f"Chatbot: {model_response}")
        print("-" * 30)

        get_chunks = input("Would you like to see fitted chunks? ")
        if get_chunks.strip() == "yes":
            for idx, chu in enumerate(chatbot.best_chunks):
                print(f"Chunk {idx+1}: ")
                print(chu)

if __name__ == '__main__':
    main()
