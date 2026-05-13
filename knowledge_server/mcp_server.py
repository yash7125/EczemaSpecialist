import json
import os
import numpy as np
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("H&R Knowledge Base Server")

# Get the path to the knowledge base JSON file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base_H&R.json")

# Lazy loading variables for semantic search
_embed_model = None
_criteria_docs = None
_criteria_embeddings = None

def _init_semantic_search():
    global _embed_model, _criteria_docs, _criteria_embeddings
    if _embed_model is not None:
        return
        
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading S-PubMedBert-MS-MARCO embedding model...")
        _embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    except ImportError:
        raise ImportError("Please install sentence-transformers to use semantic search: pip install sentence-transformers")
        
    kb = load_kb()
    if "error" in kb:
        raise ValueError(f"Failed to load KB for embeddings: {kb['error']}")
        
    _criteria_docs = []
    
    # Flatten major and minor criteria into a searchable list
    for key, desc in kb.get("major_criteria", {}).items():
        _criteria_docs.append({"type": "Major", "name": key, "description": desc})
        
    for key, desc in kb.get("minor_criteria", {}).items():
        _criteria_docs.append({"type": "Minor", "name": key, "description": desc})
        
    # Create embeddings for all criteria based on their name + description
    texts_to_embed = [f"{doc['name']}: {doc['description']}" for doc in _criteria_docs]
    print(f"Creating embeddings for {len(texts_to_embed)} criteria...")
    _criteria_embeddings = _embed_model.encode(texts_to_embed)
    print("Semantic search initialized successfully.")

def load_kb():
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_major_criteria() -> str:
    """Retrieve the Major Criteria definitions from the Hanifin & Rajka knowledge base."""
    kb = load_kb()
    if "error" in kb:
        return f"Error loading knowledge base: {kb['error']}"
    return json.dumps(kb.get("major_criteria", {}), indent=2)

@mcp.tool()
def get_minor_criteria() -> str:
    """Retrieve the Minor Criteria definitions from the Hanifin & Rajka knowledge base."""
    kb = load_kb()
    if "error" in kb:
        return f"Error loading knowledge base: {kb['error']}"
    return json.dumps(kb.get("minor_criteria", {}), indent=2)

@mcp.tool()
def get_grading_rules() -> str:
    """Retrieve the grading rules required for a definitive Atopic Dermatitis diagnosis."""
    kb = load_kb()
    if "error" in kb:
        return f"Error loading knowledge base: {kb['error']}"
    return json.dumps(kb.get("grading_rule", "No grading rules found."), indent=2)

@mcp.tool()
def search_criteria(query: str, top_k: int = 3) -> str:
    """Perform a semantic search for criteria using a medical domain embedding model."""
    try:
        _init_semantic_search()
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Embed the incoming query
        query_embedding = _embed_model.encode([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, _criteria_embeddings)[0]
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = {}
        for idx in top_indices:
            doc = _criteria_docs[idx]
            score = float(similarities[idx])
            if score > 0.3:  # Only return somewhat relevant results
                result_key = f"{doc['type']}: {doc['name']}"
                results[result_key] = {
                    "description": doc["description"],
                    "similarity_score": round(score, 4)
                }
                
        if not results:
            return f"No highly relevant criteria found for '{query}'."
            
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return f"Semantic Search Error: {str(e)}"

if __name__ == "__main__":
    import sys
    # In Jupyter/Colab, stdio transport fails because sys.stderr lacks fileno().
    # Use --sse flag to run as an HTTP server instead.
    if "--sse" in sys.argv:
        mcp.run(transport="sse", host="127.0.0.1", port=8002)
    else:
        mcp.run(transport="stdio")
