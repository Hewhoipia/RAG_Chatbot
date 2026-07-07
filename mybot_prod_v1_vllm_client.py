# Access with: http://localhost:8010

import asyncio
from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks, status
from fastapi.responses import Response, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import Optional

from mini_func.auto_suggest.auto_suggest import get_autocomplete_suggestions

try:
    import orjson
    from fastapi.responses import ORJSONResponse
    DEFAULT_RESPONSE_CLASS = ORJSONResponse
except ImportError:
    from fastapi.responses import JSONResponse
    DEFAULT_RESPONSE_CLASS = JSONResponse

app = FastAPI(
    title="RAG Chatbot API v1 - no model",
    description="Production RAG v7 - 1 concurrent user, FP16 reranker, Redis cache",
    version="1.0",
    lifespan=None,
    default_response_class=DEFAULT_RESPONSE_CLASS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    pass

# Template
try:
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    pass

# --- MAIN PAGE ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Endpoint này sẽ được gọi khi người dùng truy cập http://localhost:8010/
    
    Nó sẽ tìm và trả về file index.html nằm trong thư mục 'templates'
    """
    return templates.TemplateResponse("index.html", {"request": request})

# API ENDPOINT FOR AUTO SUGGESTION
@app.get("/api/suggest")
async def autocomplete_endpoint(
    q: Optional[str] = Query(None, description="Ký tự người dùng đang gõ")
):
    if not q or not q.strip():
        return {"suggestions": []}
        
    try:
        suggestions = await get_autocomplete_suggestions(q)
        return {"suggestions": suggestions}
    except Exception as e:
        # Quan trọng: Print lỗi ra console để debug thay vì chỉ văng 500
        print(f"Error when suggest: {e}") 
        return {"suggestions": []}

# MAIN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mybot_prod_v1_vllm_client:app",
        host="localhost",
        port=8010,
        reload=True, # True in dev enviroment
        workers=1,
        log_level="info",
    )