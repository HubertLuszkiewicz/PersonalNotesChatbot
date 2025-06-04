import psycopg2
from pgvector.psycopg2 import register_vector
import os
from dotenv import load_dotenv

class Database:
    def __init__(self):
        load_dotenv()

        dbname = os.getenv("DB_NAME")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = int(os.getenv("DB_PORT"))
        self.conn = psycopg2.connect(
            dbname=dbname, user=user, host=host, 
            port=port, password=password
        )
        cur = self.conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self.conn)
        cur.close()
    
    def create_vector_table(self):
        cur = self.conn.cursor()
        cur.execute("""--sql
            CREATE TABLE IF NOT EXISTS vectors (
                id SERIAL PRIMARY KEY,
                chunk TEXT,
                embedding vector(768)
            );"""
        )
        self.conn.commit()
        cur.close()
    
    def write_vector(self, chunk, vector):
        cur = self.conn.cursor()
        cur.execute("""--sql 
            INSERT INTO vectors (chunk, embedding) 
            VALUES (%s, %s);""", 
            (chunk, vector)
        )
        self.conn.commit()
        cur.close()
    
    def get_best_chunks(self, question_embedding, n=5):
        cur = self.conn.cursor()
        cur.execute("""--sql 
            SELECT * 
            FROM vectors 
            ORDER BY embedding <=> %s 
            LIMIT %s;""", 
            (question_embedding, n)
        )
        results = cur.fetchall()
        best_chunks = [row[1] for row in results]

        cur.close()

        return best_chunks
    
    def delete_all_vectors(self):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM vectors;")
        self.conn.commit()
        cur.close()

    def close(self):
        if self.conn:
            self.conn.close()
