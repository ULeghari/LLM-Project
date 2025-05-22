from fastapi import FastAPI, Request, WebSocket, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import WebSocketDisconnect
from main import get_response
from vector import add_documents_to_store
from langchain_core.documents import Document
from document_processor import process_uploaded_file
from datetime import datetime
import time
import os
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        if not data.get("question"):
            raise HTTPException(status_code=400, detail="Question is required")
        
        response = get_response(data["question"])
        return JSONResponse({"response": response})
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        processed_doc = await process_uploaded_file(file)
        if not processed_doc:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        num_chunks = add_documents_to_store([processed_doc])
        
        return JSONResponse({
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": num_chunks
        })
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing file")

from fastapi import WebSocketException
from starlette.websockets import WebSocketDisconnect

@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            
            # Rate limiting
            if not hasattr(websocket, 'last_request'):
                websocket.last_request = time.time()
            else:
                if time.time() - websocket.last_request < 1:  # 1 second between messages
                    raise WebSocketException(code=1008, reason="Rate limit exceeded")
                websocket.last_request = time.time()
            
            # Message length check
            if len(question) > 500:
                await websocket.send_text("Please keep questions under 500 characters")
                continue
                
            try:
                response = get_response(question)
                await websocket.send_text(response)
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                await websocket.send_text("Sorry, I encountered an error processing your question")
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except WebSocketException as e:
        logger.warning(f"WebSocket security exception: {str(e)}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)