import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
from agents import PaperReviewAgent

# Load environment variables
load_dotenv()

app = FastAPI(title="Science Arena Challenge Example Submission")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for non-streaming endpoints
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )


@app.post("/paper_review")
async def paper_review(request: Request):
    """
    Paper review endpoint - uses BaselineAgent for review
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        paper_json = body.get("paper_json")
        model = body.get("model")
        reasoning_model = body.get("reasoning_model")
        embedding_model = body.get("embedding_model")
        enable_mm = bool(body.get("enable_mm", False))

        if paper_json is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "paper_json is required"}
            )

        if not model:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "model is required"}
            )

        agent = PaperReviewAgent(
            model=model,
            reasoning_model=reasoning_model,
            embedding_model=embedding_model,
        )

        async def generate():
            async for chunk in agent.run(
                paper_json=paper_json,
                query=query,
                enable_mm=enable_mm,
            ):
                yield chunk

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
# python -c "from dotenv import load_dotenv; load_dotenv(); import runpy; runpy.run_module('app', run_name='__main__')"