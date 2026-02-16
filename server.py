import json
import logging
from pathlib import Path
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from agent import NL2SQLAgent
from utils.llm_client import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NL2SQL Agent API")

STATIC_DIR = Path(__file__).parent / "static"

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent (loads model once at startup)
logger.info("Initializing Agent...")
llm_client = LLMClient(model_provider="local", model_name="meta-llama/Llama-3.2-3B-Instruct")
agent = NL2SQLAgent(llm_client=llm_client, db_path="data/ecommerce.db", max_retries=3)
logger.info("Agent Initialized.")


@app.get("/stream")
async def stream_query(request: Request, query: str = Query(..., description="User query")):
    """
    Stream agent progress and results using Server-Sent Events (SSE).
    Uses a sync generator so sse-starlette runs it in a thread pool,
    keeping the event loop free to flush events during blocking LLM calls.
    """
    logger.info(f"Received stream request for query: {query}")

    def event_generator():
        for event_data in agent.process_query(query):
            yield {
                "event": "message",
                "id": event_data.get("event"),
                "data": json.dumps(event_data),
            }

    return EventSourceResponse(event_generator())


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    """Serve the UI (no-cache to always get latest)."""
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


# Mount static files (for any future CSS/JS assets)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("shutdown")
def shutdown():
    """Clean up resources on shutdown."""
    if hasattr(agent, 'db_manager'):
        agent.db_manager.close()
    logger.info("Server shutdown complete.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
