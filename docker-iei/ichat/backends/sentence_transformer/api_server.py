import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any
import logging

# Assume that the execution environment is configured so that 'ichat' is a package.
try:
    from .args import parse_sentence_server_args
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure sentence-transformers is installed and the script is run as a module.")
    exit(1)

# --- Pydantic Models for API Validation ---
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = Field(default=None, description="The model name to use for the embedding.")

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: str = Field(default=None, description="The model name to use for reranking.")
    top_n: int = Field(default=None, description="The number of top results to return.")

# --- Global State ---
class ServerState:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.task_type = None

state = ServerState()
app = FastAPI()

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

def setup_routes():
    """Dynamically add routes based on the task type."""
    if state.task_type == "embedding":
        @app.post("/v1/embeddings")
        async def create_embeddings(request: EmbeddingRequest):
            if not isinstance(state.model, SentenceTransformer):
                raise HTTPException(status_code=500, detail="Embedding model not loaded correctly.")
            
            sentences = [request.input] if isinstance(request.input, str) else request.input
            
            try:
                embeddings = state.model.encode(sentences, normalize_embeddings=True)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during embedding: {str(e)}")

            data = []
            for i, emb in enumerate(embeddings):
                data.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": emb.tolist()
                })

            return {
                "object": "list",
                "data": data,
                "model": state.model_name,
                "usage": {"prompt_tokens": 0, "total_tokens": 0}
            }

    elif state.task_type == "rerank":
        @app.post("/v1/rerank")
        async def create_rerank(request: RerankRequest):
            if not isinstance(state.model, CrossEncoder):
                raise HTTPException(status_code=500, detail="Rerank model not loaded correctly.")

            pairs = [[request.query, doc] for doc in request.documents]
            
            try:
                scores = state.model.predict(pairs)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during reranking: {str(e)}")

            results = []
            for i, score in enumerate(scores):
                results.append({
                    "index": i,
                    "relevance_score": float(score),
                    "document": {"text": request.documents[i]}
                })
            
            # Sort by score in descending order
            results.sort(key=lambda x: x["relevance_score"], reverse=True)

            if request.top_n:
                results = results[:request.top_n]

            return {
                "id": f"rerank-{request.model}",
                "results": results,
                "model": state.model_name,
                "usage": {"total_tokens": 0}
            }

def main():
    """Main entry point to start the API server."""
    args = parse_sentence_server_args()

    # Setup logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(__name__)

    state.model_name = args.model_name or args.model_path.split('/')[-1]
    state.task_type = args.task_type
    
    logger.info(f"Loading model '{args.model_path}' for task '{args.task_type}' on device '{args.device}'.")

    try:
        if args.task_type == "embedding":
            state.model = SentenceTransformer(args.model_path, device=args.device)
        elif args.task_type == "rerank":
            state.model = CrossEncoder(args.model_path, device=args.device)
        else:
            raise ValueError(f"Unsupported task type: {args.task_type}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        exit(1)

    logger.info("Model loaded successfully.")
    
    # Add routes to the app
    setup_routes()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main() 