import json
import os
import numpy as np
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
# --- THE FIX: Initialize FastMCP server with host and port here! ---
mcp = FastMCP("H&R Knowledge Base Server", host="127.0.0.1", port=8002)

# Get the path to the knowledge base JSON file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base_H&R.json")

# Lazy loading variables for semantic search
_embed_tokenizer = None
_embed_model = None
_criteria_docs = None
_criteria_embeddings = None
_criterion_profiles = None
_specialist_profiles = None

SEMANTIC_QUERY_TEMPLATES = {
    "pruritus": {
        "positive": [
            "The patient has severe itching or pruritus and scratching is a major symptom.",
            "The rash is very itchy and may interfere with sleep or daily life.",
        ],
        "negative": [
            "The skin eruption is not itchy and there is no pruritus.",
        ],
    },
    "chronic_relapsing": {
        "positive": [
            "The dermatitis has been persistent or relapsing for months or years.",
            "The rash keeps coming back and has a chronic course.",
        ],
        "negative": [
            "This is a first brief isolated episode without chronic recurrence.",
        ],
    },
    "typical_distribution": {
        "positive": [
            "The dermatitis involves flexural areas such as elbow folds, behind knees, neck, wrists, face, or hands.",
            "The rash has a typical atopic dermatitis distribution pattern.",
        ],
        "negative": [
            "The rash distribution is not typical for atopic dermatitis.",
        ],
    },
    "atopy_history": {
        "positive": [
            "The patient or family has atopic dermatitis, asthma, allergic rhinitis, or hay fever.",
            "There is a personal or family atopic history.",
        ],
        "negative": [
            "There is no personal or family history of eczema, asthma, or allergic rhinitis.",
        ],
    },
    "xerosis": {
        "positive": [
            "The patient has generalized dry skin or persistently sensitive dry skin even between flares.",
            "There is baseline xerosis.",
        ],
        "negative": [
            "The patient does not have chronic dry skin.",
        ],
    },
    "childhood_onset": {
        "positive": [
            "The skin disease began in infancy or early childhood.",
            "Symptoms started before school age.",
        ],
        "negative": [
            "The skin disease started only in adulthood.",
        ],
    },
}

SPECIALIST_QUERY_TEMPLATES = {
    "morphology_support": {
        "description": (
            "Typical atopic dermatitis morphology includes eczematous erythematous patches or plaques, "
            "excoriation, scaling, crusting, lichenification, fissuring, or xerotic inflammatory change."
        ),
        "positive": [
            "The lesion morphology is compatible with eczematous atopic dermatitis, with excoriation, erythema, scale, crust, or lichenification.",
            "The visible lesion pattern supports atopic dermatitis morphology.",
        ],
        "negative": [
            "The lesion morphology is not suggestive of atopic dermatitis and lacks typical eczematous features.",
        ],
    },
    "distribution_support": {
        "description": "Typical atopic dermatitis distribution favors flexural or characteristic age-appropriate sites.",
        "positive": [
            "The rash involves a distribution typical for atopic dermatitis such as flexural surfaces, neck, face, hands, or wrists.",
            "The visible distribution pattern supports atopic dermatitis.",
        ],
        "negative": [
            "The visible rash distribution is not typical for atopic dermatitis.",
        ],
    },
}

def _init_semantic_search():
    global _embed_tokenizer, _embed_model, _criteria_docs, _criteria_embeddings, _criterion_profiles, _specialist_profiles
    if (
        _embed_model is not None
        and _embed_tokenizer is not None
        and _criterion_profiles is not None
        and _specialist_profiles is not None
    ):
        return

    import torch
    from transformers import AutoModel, AutoTokenizer

    print("Loading S-PubMedBert-MS-MARCO embedding model...")
    _embed_tokenizer = AutoTokenizer.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO")
    _embed_model = AutoModel.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO")
    _embed_model.eval()
    if torch.cuda.is_available():
        _embed_model = _embed_model.to("cuda")

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
    _criteria_embeddings = _encode_texts(texts_to_embed)

    criteria_context = {
        "pruritus": kb["major_criteria"]["pruritus"],
        "chronic_relapsing": kb["major_criteria"]["chronic_history"],
        "typical_distribution": kb["major_criteria"]["distribution"],
        "atopy_history": kb["major_criteria"]["atopy_history"],
        "xerosis": kb["minor_criteria"]["xerosis"],
        "childhood_onset": kb["minor_criteria"]["early_age_onset"],
    }
    _criterion_profiles = {}
    for criterion, templates in SEMANTIC_QUERY_TEMPLATES.items():
        positive_texts = [
            f"{criteria_context[criterion]} Clinical hypothesis: {text}"
            for text in templates["positive"]
        ]
        negative_texts = [
            f"{criteria_context[criterion]} Contradictory hypothesis: {text}"
            for text in templates["negative"]
        ]
        _criterion_profiles[criterion] = {
            "description": criteria_context[criterion],
            "positive_embeddings": _encode_texts(positive_texts),
            "negative_embeddings": _encode_texts(negative_texts),
        }

    specialist_context = {
        "morphology_support": SPECIALIST_QUERY_TEMPLATES["morphology_support"]["description"],
        "distribution_support": kb["major_criteria"]["distribution"],
    }
    _specialist_profiles = {}
    for criterion, templates in SPECIALIST_QUERY_TEMPLATES.items():
        positive_texts = [
            f"{specialist_context[criterion]} Clinical hypothesis: {text}"
            for text in templates["positive"]
        ]
        negative_texts = [
            f"{specialist_context[criterion]} Contradictory hypothesis: {text}"
            for text in templates["negative"]
        ]
        _specialist_profiles[criterion] = {
            "description": specialist_context[criterion],
            "positive_embeddings": _encode_texts(positive_texts),
            "negative_embeddings": _encode_texts(negative_texts),
        }
    print("Semantic search initialized successfully.")

def _encode_texts(texts: list[str]):
    import torch
    import torch.nn.functional as F

    assert _embed_model is not None
    assert _embed_tokenizer is not None

    encoded = _embed_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    device = next(_embed_model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        model_output = _embed_model(**encoded)

    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * attention_mask, dim=1)
    counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    sentence_embeddings = summed / counts
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy()

def load_kb():
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def _score_profiles(query_text: str, profiles: dict) -> dict:
    from sklearn.metrics.pairwise import cosine_similarity

    _init_semantic_search()
    query_embedding = _encode_texts([query_text])
    results = {}

    for criterion, profile in profiles.items():
        positive_scores = cosine_similarity(query_embedding, profile["positive_embeddings"])[0]
        negative_scores = cosine_similarity(query_embedding, profile["negative_embeddings"])[0]
        best_positive = float(max(positive_scores)) if len(positive_scores) else 0.0
        best_negative = float(max(negative_scores)) if len(negative_scores) else 0.0
        margin = best_positive - best_negative

        if best_positive >= 0.58 and margin >= 0.08:
            status = "met"
        elif best_negative >= 0.58 and (best_negative - best_positive) >= 0.08:
            status = "unmet"
        else:
            status = "uncertain"

        results[criterion] = {
            "status": status,
            "positive_similarity": round(best_positive, 4),
            "negative_similarity": round(best_negative, 4),
            "description": profile["description"],
        }

    return results


def _score_patient_history(patient_history: str) -> dict:
    _init_semantic_search()
    assert _criterion_profiles is not None
    return _score_profiles(patient_history, _criterion_profiles)


def _score_specialist_findings(specialist_findings: str) -> dict:
    _init_semantic_search()
    assert _specialist_profiles is not None
    return _score_profiles(specialist_findings, _specialist_profiles)

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
        query_embedding = _encode_texts([query])
        
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

@mcp.tool()
def warm_semantic_index() -> str:
    """Load and cache the PubMedBERT semantic model and criterion embeddings before inference."""
    _init_semantic_search()
    return "Semantic index warmed and ready."

@mcp.tool()
def score_patient_history(patient_history: str) -> str:
    """Semantically score patient history against Hanifin & Rajka-relevant criteria."""
    try:
        return json.dumps(_score_patient_history(patient_history), indent=2)
    except Exception as e:
        return f"Semantic Patient Scoring Error: {str(e)}"

@mcp.tool()
def score_specialist_findings(specialist_findings: str) -> str:
    """Semantically score specialist lesion findings for atopic dermatitis morphology and distribution support."""
    try:
        return json.dumps(_score_specialist_findings(specialist_findings), indent=2)
    except Exception as e:
        return f"Semantic Specialist Scoring Error: {str(e)}"

if __name__ == "__main__":
    import sys
    # In Jupyter/Colab, stdio transport fails because sys.stderr lacks fileno().
    # Use --sse flag to run as an HTTP server instead.
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
