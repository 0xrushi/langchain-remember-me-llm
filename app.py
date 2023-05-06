import sqlite3
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTVectorStoreIndex as GPTSimpleVectorIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
import torch
from langchain.llms.base import LLM
from transformers import pipeline
from llama_index import Document
from flask import Flask, jsonify, request

import logging
 
logging.getLogger().setLevel(logging.CRITICAL)

app = Flask(__name__)

class customLLM(LLM):
    model_name = "google/flan-t5-small"
    pipeline = pipeline("text2text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})
    initial_prompt = 'You are a Q&A bot, a highly intelligent question answering bot based on the information provided by the user. If the answer cannot be found in the information, write "I could not find an answer."'

    def _call(self, prompt, stop=None):
        text = f"{self.initial_prompt}\n\n{prompt} {stop}" if stop is not None else f"{self.initial_prompt}\n\n{prompt}"
        return self.pipeline(text, max_length=9999)[0]["generated_text"]
 
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"
    
# -------------DB---------------
DB_FILE = '/root/.cache/reminders.db'
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reminders
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def add_to_db(text):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO reminders (text) VALUES (?)", (text,))
    conn.commit()
    conn.close()

def get_rows_from_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT text FROM reminders")
    rows = c.fetchall()
    conn.close()
    return rows
# ----------------------------

llm_predictor = LLMPredictor(llm=customLLM())

hfemb = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(hfemb)

"""## Save your remember to questions in this text_list DB"""

# text_list = ["remember i have kept my keys in the bedroom drawer", "I need to go to shopping on saturday"]

init_db()
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

@app.route("/", methods=['GET'])
def rootpage():
    return jsonify({'response': "hello world"})

@app.route('/query', methods=['GET'])
def query():
    query = request.args.get('query', default='', type=str)
    rows = get_rows_from_db()
    documents = [Document(row[0]) for row in rows]
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    response = query_engine.query(query)
    return jsonify({'response': response.response})

@app.route('/write', methods=['POST'])
def write():
    content = request.json.get('content')
    if not content:
        return jsonify({'error': 'Content is required'}), 400
    add_to_db(content)
    return jsonify({'message': 'Content added successfully'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
