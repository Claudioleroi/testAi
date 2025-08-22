from fastapi import FastAPI, UploadFile, File, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import os
import shutil
import queue
import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional
import psutil
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ServiceClientAI")

app = FastAPI(title="Service Client AI Assistant API")

# Shared resources
file_contents: Dict[str, pd.DataFrame] = {}
company_name: Optional[str] = "MTN"
mtn_packages_data: Optional[pd.DataFrame] = None
mtn_knowledge_base: str = ""
log_queue = queue.Queue()
thread_pool = asyncio.get_event_loop()

# Google Gemini settings
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyABpwURhdtTO4A9BVv-6T41-yhvtS8X0Z8")
GEMINI_MODEL: str = "gemini-2.0-flash"

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging handler to send logs to the queue
class CustomHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "process_id": os.getpid(),
            "thread_id": record.thread,
            "memory_usage": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        }
        log_queue.put(log_entry)

logger.addHandler(CustomHandler())

# Context management for query and file data
class ContextManager:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.document_vectors = {}

    def update_vectors(self, file_name: str, content: str):
        if not self.document_vectors:
            self.vectorizer.fit([content])
        vector = self.vectorizer.transform([content])
        self.document_vectors[file_name] = vector

    def get_relevant_context(self, query: str, file_names: Optional[List[str]] = None, max_length: int = 2000) -> str:
        if not self.document_vectors:
            return ""
        query_vector = self.vectorizer.transform([query])
        relevant_parts = []

        files_to_search = file_names if file_names else self.document_vectors.keys()
        for file_name in files_to_search:
            if file_name in self.document_vectors:
                similarity = cosine_similarity(query_vector, self.document_vectors[file_name])[0][0]
                if similarity > 0.1:  # Threshold for relevance
                    content = file_contents[file_name]["text"].str.cat(sep=" ") if "text" in file_contents[file_name].columns else file_contents[file_name].to_string()
                    relevant_parts.append(content[:max_length])

        return "\n\n".join(relevant_parts)

context_manager = ContextManager()

def format_mtn_prompt(text: str, language: str, context: str) -> str:
    if language == "fr":
        mtn_intro = """Je suis MTN AI, votre assistant intelligent MTN. Je peux vous aider avec :
- Les forfaits MTN (data, voix, SMS)
- Les codes USSD pour activer vos services
- Les informations sur les prix et validités
- Les services MTN Mobile Money
- L'assistance technique MTN

Question: {text}

Contexte MTN: {context}

Répondez de manière précise et utile en tant qu'assistant MTN officiel."""
    else:
        mtn_intro = """I am MTN AI, your intelligent MTN assistant. I can help you with:
- MTN packages (data, voice, SMS)
- USSD codes to activate services
- Information about prices and validity
- MTN Mobile Money services
- MTN technical support

Question: {text}

MTN Context: {context}

Respond accurately and helpfully as an official MTN assistant."""
    
    return mtn_intro.format(text=text, context=context)

def load_mtn_knowledge():
    """Load MTN packages and knowledge base data"""
    global mtn_packages_data, mtn_knowledge_base
    
    try:
        # Construct absolute paths to the knowledge files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(base_dir, "Forfaits_MTN_Codes_USSD_2025-08-22.xlsx")
        card_path = os.path.join(base_dir, "temp_data", "card.txt")

        # Verify files exist
        if not os.path.exists(excel_path):
            logger.error(f"Excel file not found: {excel_path}")
            return False
            
        if not os.path.exists(card_path):
            logger.error(f"Card file not found: {card_path}")
            return False

        # Load MTN packages Excel file
        mtn_packages_data = pd.read_excel(excel_path)
        logger.info(f"✅ Loaded {len(mtn_packages_data)} MTN packages from {excel_path}")
        
        # Display column names for debugging
        logger.info(f"Excel columns: {list(mtn_packages_data.columns)}")
        
        # Load MTN knowledge base text
        with open(card_path, "r", encoding="utf-8") as f:
            mtn_knowledge_base = f.read()
        logger.info(f"✅ Loaded MTN knowledge base ({len(mtn_knowledge_base)} characters)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading MTN knowledge: {str(e)}")
        return False

def search_mtn_packages(query: str, country: str = None) -> str:
    """Search MTN packages based on query with improved keyword matching."""
    if mtn_packages_data is None:
        logger.warning("MTN packages data not loaded - attempting to reload")
        if not load_mtn_knowledge():
            return "❌ Base de données MTN non disponible."
    
    if mtn_packages_data is None or len(mtn_packages_data) == 0:
        return "❌ Aucune donnée MTN trouvée."

    query_lower = query.lower()
    # Extract keywords like '1 jour', '7 jours', '30 jours', 'internet', 'voix'
    keywords = re.findall(r'\b(\d+\s*jour[s]?|internet|data|voix|appel[s]?|sms)\b', query_lower)

    df = mtn_packages_data
    if country:
        df = df[df['Pays'].str.lower() == country.lower()]

    # If no specific keywords, do a broad search
    if not keywords:
        mask = (
            df['Nom Forfait'].str.lower().str.contains(query_lower, na=False) |
            df['Description'].str.lower().str.contains(query_lower, na=False)
        )
        matching_packages = df[mask]
    else:
        # Filter based on keywords
        mask = pd.Series([True] * len(df), index=df.index)
        for keyword in keywords:
            # Normalize '1 jour' to '24h' or '1 jour' for better matching in 'Validité'
            if 'jour' in keyword:
                day_search = keyword.split()[0]
                if day_search == '1':
                    period_mask = df['Validité'].str.contains('24h', case=False, na=False) | df['Validité'].str.contains('1 jour', case=False, na=False)
                else:
                    period_mask = df['Validité'].str.contains(day_search, case=False, na=False)
                mask &= period_mask
            else:
                # Search in name, description, and type for other keywords
                mask &= (
                    df['Nom Forfait'].str.lower().str.contains(keyword, na=False) |
                    df['Description'].str.lower().str.contains(keyword, na=False) |
                    df['Type Service'].str.lower().str.contains(keyword, na=False)
                )
        matching_packages = df[mask]

    results = []
    for _, package in matching_packages.head(5).iterrows():
        result = f"""
### 📱 {package['Nom Forfait']} ({package['Pays']})
- **💰 Prix:** {package['Prix (FCFA/NGN/GHS/UGX/ZAR)']}
- **📊 Volume:** {package['Volume/Minutes']}
- **⏰ Validité:** {package['Validité']}
- **📞 Code USSD:** `{package['Code USSD']}`
- **ℹ️ Description:** {package['Description']}
        """
        results.append(result.strip())

    return "\n\n".join(results) if results else ""

async def generate_gemini_response_stream(prompt: str, websocket: WebSocket):
    """Generate streaming response from Gemini and send via WebSocket"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Send typing indicator
        await websocket.send_json({"type": "typing", "content": "..."})
        
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.3,
            },
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                await websocket.send_json({
                    "type": "chunk", 
                    "content": chunk.text,
                    "full_content": full_response
                })
                await asyncio.sleep(0.05)  # Small delay for smoother streaming
        
        # Send completion signal
        await websocket.send_json({
            "type": "complete", 
            "content": full_response
        })
        
        return full_response
        
    except Exception as e:
        logger.error(f"Error generating streaming response: {str(e)}")
        await websocket.send_json({
            "type": "error", 
            "content": f"Erreur lors de la génération de la réponse: {str(e)}"
        })
        return ""

async def generate_gemini_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.3,  # Lower temperature for more consistent MTN responses
                "top_p": 0.9,
            }
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error in Gemini inference: {str(e)}")
        raise

async def generate_bedrock_response(prompt: str) -> str:
    try:
        response_text = await generate_gemini_response(prompt)
        response_text = re.sub(r'^(Réponse:|Response:)\s*', '', response_text, flags=re.IGNORECASE)
        
        # Limiter à la première phrase ou aux deux premières phrases
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        concise_response = ' '.join(sentences[:2]).strip()
        
        return concise_response if concise_response else "Je suis désolé, je ne peux pas répondre à cette question."
    except Exception as e:
        logger.error(f"Error in Bedrock inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")
user_language = {}  # Stores language preference per user session

class Query(BaseModel):
    text: str
    max_length: int = 10000
    language: str = "fr"
    file_names: Optional[List[str]] = None

@app.post("/predict")
async def predict(request: Request, query: Query):
    try:
        logger.info(f"Received MTN AI request: {query.text[:50]}...")
        
        # Get or set user's language preference
        session_id = request.client.host
        user_language[session_id] = query.language
        
        # Search MTN packages for relevant information
        mtn_context = search_mtn_packages(query.text)
        
        # Add general MTN knowledge base context
        general_context = mtn_knowledge_base[:1000] if mtn_knowledge_base else ""
        
        # Combine contexts
        full_context = f"{mtn_context}\n\n{general_context}"
        
        # If specific packages are found, present them directly.
        # Otherwise, use the general knowledge base for a broader answer.
        if mtn_context:
            prompt = f"""En tant qu'assistant MTN, réponds à la question de l'utilisateur en te basant sur les forfaits suivants que j'ai trouvés. Utilise le format markdown pour une présentation claire et organisée.

Question: {query.text}

Forfaits trouvés:
{mtn_context}

Réponds en markdown avec une structure claire incluant des titres, listes et mise en forme appropriée."""
        else:
            # Format a general prompt if no specific packages are found
            general_context = mtn_knowledge_base[:1500] if mtn_knowledge_base else ""
            prompt = format_mtn_prompt(query.text, query.language, general_context)

        # Generate response using Gemini
        response = await generate_gemini_response(prompt)
        
        # Clean up the response
        response = re.sub(r'^(Réponse:|Response:)\s*', '', response, flags=re.IGNORECASE)
        
        logger.info(f"MTN AI response generated successfully")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error in MTN AI prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-training-data")
async def upload_training_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global company_name
    try:
        logger.info(f"Receiving file upload: {file.filename}")
        os.makedirs("temp_data", exist_ok=True)
        file_path = f"temp_data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        data = await process_file_content(file_path, file.filename)
        file_contents[file.filename] = data
        text_content = data["text"].str.cat(sep=" ") if "text" in data.columns else data.to_string()
        background_tasks.add_task(context_manager.update_vectors, file.filename, text_content)
        if "text" in data.columns:
            company_name = extract_company_name(text_content) or company_name
        return {"message": "File uploaded and processed successfully", "filename": file.filename, "company_name": company_name}
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_file_content(file_path: str, file_name: str) -> pd.DataFrame:
    if file_name.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_name.endswith(".txt"):
        with open(file_path, "r") as f:
            text_content = f.read()
        return pd.DataFrame({"text": text_content.splitlines()})
    elif file_name.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        return pd.DataFrame({"text": text.splitlines()})
    else:
        raise ValueError("Unsupported file format")

def extract_company_name(text: str) -> Optional[str]:
    match = re.search(r"(Company Name|Organization|Client Name):\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    first_line = text.splitlines()[0].strip()
    if first_line:
        return first_line

@app.get("/logs")
async def stream_logs():
    async def log_stream() -> AsyncGenerator[str, None]:
        while True:
            try:
                if not log_queue.empty():
                    log_entry = log_queue.get_nowait()
                    yield f"data: {json.dumps(log_entry)}\n\n"
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in log stream: {e}")
                yield f"data: {json.dumps({'level': 'ERROR', 'message': str(e)})}\n\n"

    return StreamingResponse(log_stream(), media_type="text/event-stream")

# Initialize MTN knowledge on startup
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "chat":
                query_text = data["text"]
                language = data.get("language", "fr")
                
                logger.info(f"WebSocket MTN AI request: {query_text[:50]}...")
                
                # Search MTN packages for relevant information
                mtn_context = search_mtn_packages(query_text)
                
                # Generate prompt based on context
                if mtn_context:
                    prompt = f"""En tant qu'assistant MTN, réponds à la question de l'utilisateur en te basant sur les forfaits suivants que j'ai trouvés. Utilise le format markdown pour une présentation claire et organisée.

Question: {query_text}

Forfaits trouvés:
{mtn_context}

Réponds en markdown avec une structure claire incluant des titres, listes et mise en forme appropriée."""
                else:
                    general_context = mtn_knowledge_base[:1500] if mtn_knowledge_base else ""
                    prompt = format_mtn_prompt(query_text, language, general_context)

                # Generate streaming response
                await generate_gemini_response_stream(prompt, websocket)
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "content": f"Erreur de connexion: {str(e)}"
        })

@app.on_event("startup")
async def startup_event():
    logger.info("Starting MTN AI Assistant...")
    success = load_mtn_knowledge()
    if success:
        logger.info("✅ MTN AI Assistant ready!")
    else:
        logger.error("❌ MTN AI Assistant started with errors!")

# Run FastAPI with uvicorn
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    
    # Reload environment variables
    load_dotenv()
    
    # Get port from environment variable or use default 8000
    server_port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting MTN AI server on port {server_port}...")
    uvicorn.run(app, host="0.0.0.0", port=server_port)