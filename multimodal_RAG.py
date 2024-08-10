import requests
import json

from langchain_ollama.llms import OllamaLLM
import embed_anything
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import os
from pinecone import PineconeApiException
import uuid
from embed_anything import EmbedData
from fastapi import FastAPI
from pydantic import BaseModel
from embed_anything.vectordb import PineconeAdapter
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain.chains import LLMChain
from fastapi import FastAPI, HTTPException

from embed_anything.vectordb import PineconeAdapter


ollama = OllamaLLM(base_url='http://localhost:11434', model='llama2')




docs = embed_anything.embed_webpage(url="PUT-LINK", embeder="Jina")

pc =  Pinecone(api_key)










# Initialize the PineconeEmbedder class
api_key = os.environ.get("PINECONE_API_KEY")
index_name = "anything"
pinecone_adapter = PineconeAdapter(api_key)

try:
    pinecone_adapter.delete_index("anything")
except:
    pass

# Initialize the PineconeEmbedder class

pinecone_adapter.create_index(dimension=768, metric="cosine")



app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/ollama")
async def run_ollama(query: Query):
    embed_query = embed_anything.embed_query([query.prompt], embeder="Jina")
    result = index_name.query(
        vector=embed_query[0].embedding,
        top_k=2,
        include_metadata=True,
    )
    print(result)
    metadata_texts = [item['metadata']['text'] for item in result['matches']]
    
    joined_text = ''.join(query.prompt) + ' '.join(metadata_texts[0])
    response = ollama(joined_text)
    return response

