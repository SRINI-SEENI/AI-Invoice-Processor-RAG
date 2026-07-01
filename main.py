import os
import shutil
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import RAG system functions
from rag_system import read_and_chunk_file, create_index, retrieve_top_chunks, generate_answer, run_query_on_pdf

app = FastAPI(title="Invoice Processing RAG System")

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Cache for loaded PDF states: pdf_path -> { "chunks": chunks, "index": index }
pdf_cache = {}

class QueryRequest(BaseModel):
    pdf_path: str
    query: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process and build index
        print(f"Processing uploaded file: {file.filename}")
        chunks = read_and_chunk_file(file_path)
        index = create_index(chunks)
        
        # Save to cache
        pdf_cache[file_path] = {
            "chunks": chunks,
            "index": index
        }
        
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "filepath": file_path,
            "chunk_count": len(chunks),
            "preview_chunks": chunks[:3] if chunks else []
        })
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/api/query")
async def query_invoice(request: QueryRequest):
    pdf_path = request.pdf_path
    query = request.query
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Selected invoice PDF file not found.")
        
    try:
        # Check if we have the index in cache, otherwise create it
        if pdf_path in pdf_cache:
            cache = pdf_cache[pdf_path]
            chunks = cache["chunks"]
            index = cache["index"]
        else:
            chunks = read_and_chunk_file(pdf_path)
            index = create_index(chunks)
            pdf_cache[pdf_path] = {
                "chunks": chunks,
                "index": index
            }
            
        # Run step-by-step to demonstrate LangGraph flow in frontend
        # Step 1: Loaded chunks count (already done)
        # Step 2: Retrieve chunks
        retrieved_chunks = retrieve_top_chunks(index, query)
        
        # Step 3: Generate answer
        answer = generate_answer(query, retrieved_chunks)
        
        # We can construct the visual execution path
        execution_path = [
            {"node": "START", "status": "completed", "output": "Graph started."},
            {"node": "load_pdf", "status": "completed", "output": f"Loaded PDF. Extracted {len(chunks)} chunks."},
            {"node": "retrieve", "status": "completed", "output": f"Retrieved top {len(retrieved_chunks)} relevant contexts."},
            {"node": "generate", "status": "completed", "output": "Answer generated via LangChain LLM / Extractor."},
            {"node": "END", "status": "completed", "output": "Graph finished."}
        ]
        
        return JSONResponse(content={
            "status": "success",
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "execution_path": execution_path
        })
    except Exception as e:
        print(f"Error querying invoice: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
