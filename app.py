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


llm_predictor = LLMPredictor(llm=customLLM())

hfemb = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(hfemb)

"""## Save your remember to questions in this text_list DB"""

text_list = ["remember i have kept my keys in the bedroom drawer", "I need to go to shopping on saturday"]

documents = [Document(t) for t in text_list]

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

@app.route('/query', methods=['GET'])
def query():
    query = request.args.get('query', default='', type=str)
    response = query_engine.query(query)
    return jsonify({'response': response.response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
