import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.super_graph import super_graph


app = FastAPI(title="Content Intelligence Pipeline API")

class QueryRequest(BaseModel):
    request: str

@app.post("/generate-insight")
def generate_report(payload: QueryRequest):
    try:
        response = super_graph.invoke({
            "messages": [
                ("user", payload.request)
            ]},
            {"recursion_limit": 20}
        )
        return {"report": response['report']}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "POST JSON {'request': 'your prompt'} to /generate-report to retrieve a report."
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
