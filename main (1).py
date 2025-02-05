from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["*"]
)

# OpenAI API client
client = OpenAI(api_key="your_openai_api_key")

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: List[str]

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/similarity", response_model=SimilarityResponse)
def get_most_similar(request: SimilarityRequest):
    try:
        # Compute embeddings for docs
        doc_embeddings = [get_embedding(doc) for doc in request.docs]
        # Compute embedding for query
        query_embedding = get_embedding(request.query)
        
        # Compute similarities
        similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
        
        # Rank documents by similarity
        sorted_indices = np.argsort(similarities)[::-1][:3]
        sorted_docs = [request.docs[i] for i in sorted_indices]
        
        return {"matches": sorted_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example local endpoint URL: http://127.0.0.1:8000/similarity
