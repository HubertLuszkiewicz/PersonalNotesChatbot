# Personal Notes Chatbot

This project is my implementation of RAG chatbot which goal is to retrieve information from your personal notes! It uses Google Gemini API which at the time of building this app is free to use! (at least some models are)

## Getting Started

### Setup database
1. Create and start container with PostgreSQL + pgvector database
   
    ```
    docker compose up -d
    ```

### Setup chatbot
1. Generate Gemini API key here: https://aistudio.google.com/apikey
2. Clone this repository
   ```bash
   git clone https://github.com/HubertLuszkiewicz/PersonalNotesChatbot.git
   ```
4. Install required packages
   ```bash
   pip install google-genai psycopg2 pgvector python-dotenv numpy
   ```
   
6. In project root directory add .env file with following content:
   ```dotenv
   API_KEY=your_gemini_api_key
   DB_NAME=mydatabase
   DB_USER=myuser
   DB_PASSWORD=mypassword
   DB_HOST=localhost
   DB_PORT=5431
   ```
   
7. Build knowledge base composed of .txt files which content you're interested in. Replace rainforest.txt with path to your file or use it as deafult.
   ```bash
   python .\embed_file.py .\rainforest.txt 200 30
   ```
   
## Usage
1. Run chatbot

   ```bash
   python .\chatbot_cli.py
   ```
2. Ask any question regarding your note content.
