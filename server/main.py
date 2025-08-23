from fastapi import FastAPI, UploadFile, File, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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

app = FastAPI(title="MTN AI Assistant API")

# Shared resources
file_contents: Dict[str, str] = {}
company_name: Optional[str] = "MTN"
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
                    content = file_contents[file_name]
                    relevant_parts.append(content[:max_length])

        return "\n\n".join(relevant_parts)

context_manager = ContextManager()

# Conversation memory and intelligence storage
user_language = {}  # Stores language preference per user session
user_conversations = {}  # Stores conversation history per user session
user_preferences = {}  # Stores user preferences and patterns

def detect_language(text: str) -> str:
    """Détecte automatiquement la langue de la requête"""
    french_words = ['bonjour', 'merci', 'forfait', 'internet', 'appel', 'sms', 'prix', 'code', 'usd', 'combien']
    english_words = ['hello', 'thanks', 'package', 'internet', 'call', 'sms', 'price', 'code', 'ussd', 'how much']
    
    french_count = sum(1 for word in french_words if word in text.lower())
    english_count = sum(1 for word in english_words if word in text.lower())
    
    return "fr" if french_count > english_count else "en"


def format_mtn_prompt(text: str, language: str, context: str, conversation_history: str = "") -> str:
    """Enhanced prompt with advanced intelligence and context awareness."""
    
    # Analyze the query to provide targeted instructions
    intent = analyze_query_intent(text)
    
    if language == "fr":
        # Build dynamic instructions based on intent
        specific_instructions = ""
        if intent['question_type'] == 'how_to':
            specific_instructions = "\n🔧 FOCUS: Explique étape par étape la procédure d'activation avec les codes USSD exacts."
        elif intent['question_type'] == 'pricing':
            specific_instructions = "\n💰 FOCUS: Mets l'accent sur les prix exacts en FCFA et compare les options disponibles."
        elif intent['question_type'] == 'activation_code':
            specific_instructions = "\n📱 FOCUS: Fournis les codes USSD exacts à composer, avec la syntaxe complète."
        elif intent['question_type'] == 'recommendation':
            specific_instructions = "\n🎯 FOCUS: Analyse les besoins et recommande les meilleures options avec justification."
        
        urgency_note = ""
        if intent['urgency'] == 'high':
            urgency_note = "\n⚡ URGENT: Réponds de manière concise et directe avec l'information essentielle d'abord."
        
        mtn_intro = f"""Tu es MTN AI, l'assistant expert MTN avec une intelligence contextuelle avancée.

🧠 CAPACITÉS COGNITIVES:
1. Analyse automatique de l'intention utilisateur
2. Compréhension des besoins implicites et contextuels
3. Adaptation dynamique selon l'historique de conversation
4. Recommandations personnalisées intelligentes
5. Explication claire des procédures complexes
6. Détection des urgences et priorités

📋 RÈGLES DE FONCTIONNEMENT:
- Utilise EXCLUSIVEMENT ta base de connaissances MTN ci-dessous
- Analyse l'intention derrière chaque question
- Réponds avec précision (prix exacts, codes USSD complets, durées)
- Propose des alternatives pertinentes quand approprié
- Structure tes réponses en markdown professionnel
- Adapte ton niveau de détail selon le contexte{specific_instructions}{urgency_note}

💬 HISTORIQUE DE CONVERSATION:
{conversation_history}

❓ QUESTION UTILISATEUR:
{text}

📚 BASE DE CONNAISSANCES MTN:
{context}

🎯 MISSION: Analyse l'intention, comprends le contexte, et fournis une réponse experte complète basée uniquement sur ta base de connaissances. Sois intelligent, précis et orienté solution."""
    else:
        mtn_intro = f"""I am MTN AI, your advanced intelligent MTN assistant with contextual understanding.

🧠 COGNITIVE ABILITIES:
1. Automatic user intent analysis
2. Understanding of implicit needs and context
3. Dynamic adaptation based on conversation history
4. Intelligent personalized recommendations
5. Clear explanation of complex procedures

📋 OPERATING RULES:
- Use EXCLUSIVELY your MTN knowledge base below
- Analyze the intention behind each question
- Respond with precision (exact prices, complete USSD codes, durations)
- Propose relevant alternatives when appropriate
- Structure responses in professional markdown

💬 CONVERSATION HISTORY:
{conversation_history}

❓ USER QUESTION:
{text}

📚 MTN KNOWLEDGE BASE:
{context}

🎯 MISSION: Analyze intent, understand context, and provide expert complete response based solely on your knowledge base."""
    
    return mtn_intro.format(text=text, context=context, conversation_history=conversation_history)

def load_mtn_knowledge():
    """Load MTN knowledge base data from card.txt only"""
    global mtn_knowledge_base
    
    try:
        # Construct absolute path to the knowledge file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        card_path = os.path.join(base_dir, "temp_data", "card.txt")
            
        if not os.path.exists(card_path):
            logger.error(f"Knowledge base file not found: {card_path}")
            return False
        
        # Load MTN knowledge base text
        with open(card_path, "r", encoding="utf-8") as f:
            mtn_knowledge_base = f.read()
        logger.info(f"✅ Loaded MTN knowledge base ({len(mtn_knowledge_base)} characters)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading MTN knowledge: {str(e)}")
        return False

def analyze_query_intent(query: str) -> dict:
    """Analyze user query to understand intent and extract key parameters."""
    query_lower = query.lower()
    intent = {
        'service_type': None,
        'price_range': None,
        'duration': None,
        'volume': None,
        'special_features': [],
        'question_type': 'general',
        'urgency': 'normal'
    }
    
    # Detect service type
    if any(word in query_lower for word in ['internet', 'data', 'web', 'navigation', 'connexion', 'mb', 'gb', 'ko', 'mega', 'giga']):
        intent['service_type'] = 'internet'
    elif any(word in query_lower for word in ['sms', 'message', 'texto', 'msg']):
        intent['service_type'] = 'sms'
    elif any(word in query_lower for word in ['appel', 'call', 'voix', 'voice', 'minute', 'communication']):
        intent['service_type'] = 'appel'
    elif any(word in query_lower for word in ['combo', 'pack', 'bundle', 'hybride', 'mix']):
        intent['service_type'] = 'hybrid'
    
    # Detect question type
    if any(word in query_lower for word in ['comment', 'how', 'procédure', 'étapes', 'activer']):
        intent['question_type'] = 'how_to'
    elif any(word in query_lower for word in ['combien', 'prix', 'coût', 'tarif', 'how much', 'cost']):
        intent['question_type'] = 'pricing'
    elif any(word in query_lower for word in ['code', 'ussd', 'numéro', 'composer']):
        intent['question_type'] = 'activation_code'
    elif any(word in query_lower for word in ['meilleur', 'recommande', 'conseille', 'best', 'recommend']):
        intent['question_type'] = 'recommendation'
    
    # Detect special features
    if any(word in query_lower for word in ['whatsapp', 'facebook', 'instagram', 'tiktok', 'social']):
        intent['special_features'].append('social')
    if any(word in query_lower for word in ['jeune', 'widge', 'youth', 'student', 'étudiant']):
        intent['special_features'].append('youth')
    if any(word in query_lower for word in ['weekend', 'week-end', 'samedi', 'dimanche']):
        intent['special_features'].append('weekend')
    if any(word in query_lower for word in ['nuit', 'night', 'nocturne']):
        intent['special_features'].append('night')
    
    # Extract price range
    import re
    price_match = re.search(r'(\d+)\s*(fcfa|f|franc)', query_lower)
    if price_match:
        intent['price_range'] = int(price_match.group(1))
    
    # Extract volume
    volume_match = re.search(r'(\d+)\s*(mo|go|gb|mb|ko)', query_lower)
    if volume_match:
        intent['volume'] = f"{volume_match.group(1)}{volume_match.group(2)}"
    
    # Detect urgency
    if any(word in query_lower for word in ['urgent', 'rapidement', 'vite', 'maintenant', 'immédiatement']):
        intent['urgency'] = 'high'
    
    return intent

def search_mtn_knowledge(query: str) -> str:
    """Enhanced intelligent search with intent analysis and contextual understanding."""
    if not mtn_knowledge_base:
        logger.warning("MTN knowledge base not loaded - attempting to reload")
        if not load_mtn_knowledge():
            return "❌ Base de connaissances MTN non disponible."
    
    query_lower = query.lower()
    lines = mtn_knowledge_base.split('\n')
    
    # Analyze query intent
    intent = analyze_query_intent(query)
    
    # Enhanced keyword mapping with context awareness
    keyword_mapping = {
        'forfait': ['forfait', 'package', 'plan', 'abonnement', 'offre', 'bundle', 'service'],
        'internet': ['internet', 'data', 'web', 'navigation', 'connexion', 'mb', 'gb', 'ko', 'mega', 'giga', 'net', 'wifi'],
        'sms': ['sms', 'message', 'texto', 'msg', 'messages', 'texto'],
        'appel': ['appel', 'call', 'voix', 'voice', 'minute', 'min', 'talk', 'communication', 'téléphone'],
        'code': ['code', 'ussd', 'activation', 'composer', 'numéro', 'dial', '*', '#'],
        'prix': ['prix', 'price', 'coût', 'fcfa', 'tarif', 'montant', 'argent', 'payer', 'gratuit'],
        'durée': ['durée', 'validité', 'temps', 'jour', 'jours', 'semaine', 'mois', 'heure', 'h', 'expire'],
        'social': ['whatsapp', 'facebook', 'instagram', 'tiktok', 'social', 'réseau', 'wa', 'fb', 'ig'],
        'jeune': ['jeune', 'widge', 'youth', 'student', 'étudiant', 'école', 'université'],
        'weekend': ['weekend', 'week-end', 'samedi', 'dimanche', 'vsd', 'fin de semaine'],
        'nuit': ['nuit', 'night', 'nocturne', '23h', '6h', 'minuit', 'soir'],
        'hybrid': ['hybrid', 'hybride', 'mix', 'combo', 'combiné', 'tout-en-un'],
        'partage': ['partage', 'share', 'famille', 'group', 'groupe', 'ami'],
        'activation': ['activer', 'activate', 'comment', 'procédure', 'étapes', 'how'],
        'mobile_money': ['mobile money', 'momo', 'transfert', 'argent', 'paiement', 'transaction']
    }
    
    # Detect categories with intent-based weighting
    detected_categories = []
    category_weights = {}
    
    for category, keywords in keyword_mapping.items():
        weight = 0
        for keyword in keywords:
            if keyword in query_lower:
                # Higher weight for exact service type matches
                if category == intent['service_type']:
                    weight += 3
                elif category in intent['special_features']:
                    weight += 2
                else:
                    weight += 1
        
        if weight > 0:
            detected_categories.append(category)
            category_weights[category] = weight
    
    # Smart section extraction with intent awareness
    relevant_sections = []
    current_section = []
    in_table = False
    section_relevance_score = 0
    context_buffer = []  # Store context around relevant sections
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        line_relevance = 0
        
        # Calculate line relevance with intent-based scoring
        for category in detected_categories:
            category_weight = category_weights.get(category, 1)
            for keyword in keyword_mapping[category]:
                if keyword in line_lower:
                    line_relevance += category_weight
        
        # Special scoring for tables and headers
        if '|' in line:
            if any(keyword in line_lower for keyword in ['forfait', 'code', 'prix', 'ussd', 'volume']):
                line_relevance += 4
                in_table = True
            elif in_table:
                line_relevance += 3
        elif line.startswith('#'):
            header_relevance = sum(2 for cat in detected_categories if any(kw in line_lower for kw in keyword_mapping[cat]))
            line_relevance += header_relevance + 2
        
        # Intent-specific boosting
        if intent['question_type'] == 'activation_code' and any(code_word in line_lower for code_word in ['*', '#', 'composer', 'dial']):
            line_relevance += 3
        elif intent['question_type'] == 'pricing' and any(price_word in line_lower for price_word in ['fcfa', 'prix', 'gratuit', 'coût']):
            line_relevance += 3
        elif intent['question_type'] == 'how_to' and any(how_word in line_lower for how_word in ['comment', 'étapes', 'procédure', 'activer']):
            line_relevance += 3
        
        # Include relevant lines with context
        if line_relevance > 0:
            # Add context from previous lines if starting new section
            if not current_section and i > 0:
                for j in range(max(0, i-2), i):
                    if lines[j].strip() and not lines[j] in current_section:
                        current_section.append(lines[j])
            
            current_section.append(line)
            section_relevance_score += line_relevance
            
            # Add context from next lines for tables
            if in_table and i < len(lines) - 1:
                next_line = lines[i + 1]
                if '|' in next_line and next_line not in current_section:
                    current_section.append(next_line)
        
        elif current_section and (line.strip() == '' or (not in_table and line_relevance == 0)):
            if section_relevance_score > 1:  # Lower threshold for better coverage
                relevant_sections.extend(current_section)
                relevant_sections.append('')  # Add separator
            current_section = []
            section_relevance_score = 0
            in_table = False
    
    # Add final section if relevant
    if current_section and section_relevance_score > 1:
        relevant_sections.extend(current_section)
    
    # Enhanced fallback with intent-based search
    if not relevant_sections or len(relevant_sections) < 5:
        logger.info(f"Using fallback search for query: {query}")
        words = [w for w in query_lower.split() if len(w) > 2]
        fallback_sections = []
        
        # Priority search based on intent
        priority_keywords = []
        if intent['service_type']:
            priority_keywords.extend(keyword_mapping.get(intent['service_type'], []))
        if intent['question_type'] == 'activation_code':
            priority_keywords.extend(['code', 'ussd', '*', '#', 'composer'])
        elif intent['question_type'] == 'pricing':
            priority_keywords.extend(['prix', 'fcfa', 'coût', 'gratuit'])
        
        for line in lines:
            line_lower = line.lower()
            
            # Score line based on word matches and priority
            word_score = sum(1 for word in words if word in line_lower)
            priority_score = sum(2 for keyword in priority_keywords if keyword in line_lower)
            table_score = 2 if '|' in line else 0
            header_score = 1 if line.startswith('#') else 0
            
            total_score = word_score + priority_score + table_score + header_score
            
            if total_score > 0:
                fallback_sections.append((line, total_score))
        
        # Sort by score and take top results
        fallback_sections.sort(key=lambda x: x[1], reverse=True)
        selected_lines = [line for line, score in fallback_sections[:150]]
        
        if selected_lines:
            return '\n'.join(selected_lines)
        else:
            # Last resort: return general MTN information
            return mtn_knowledge_base[:3000]
    
    result = '\n'.join(relevant_sections)
    logger.info(f"Found {len(relevant_sections)} relevant lines for query: {query[:50]}...")
    return result

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
                "temperature": 0.3,
                "top_p": 0.9,
            }
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error in Gemini inference: {str(e)}")
        raise

class Query(BaseModel):
    text: str
    max_length: int = 10000
    language: str = "fr"
    file_names: Optional[List[str]] = None

@app.post("/predict")
async def predict(request: Request, query: Query):
    global user_language, user_conversations
    
    client_ip = request.client.host
    
    # Détection automatique de la langue si non spécifiée
    if not query.language:
        query.language = detect_language(query.text)
    user_language[client_ip] = query.language
    
    if client_ip not in user_conversations:
        user_conversations[client_ip] = []
    
    try:
        logger.info(f"Received MTN AI request: {query.text}")
        
        # Search MTN knowledge base for relevant information
        mtn_context = search_mtn_knowledge(query.text)
        
        # Préparation de l'historique de conversation
        conversation_history = ""
        if user_conversations[client_ip]:
            recent_history = user_conversations[client_ip][-10:]
            conversation_history = "\n".join([
                f"{'Utilisateur' if i % 2 == 0 else 'Assistant'}: {msg}" 
                for i, msg in enumerate(recent_history)
            ])
        
        # Format the prompt with MTN context
        prompt = format_mtn_prompt(query.text, query.language, mtn_context, conversation_history)
        
        # Generate response using Gemini
        response = await generate_gemini_response(prompt)
        
        # Clean up the response
        response = re.sub(r'^(Réponse:|Response:)\s*', '', response, flags=re.IGNORECASE)
        
        # Mise à jour de l'historique
        user_conversations[client_ip].append(query.text)
        user_conversations[client_ip].append(response)
        
        if len(user_conversations[client_ip]) > 20:
            user_conversations[client_ip] = user_conversations[client_ip][-20:]
        
        logger.info("MTN AI response generated successfully")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return {
            "response": "Désolé, une erreur s'est produite. Veuillez réessayer.",
            "error": str(e),
            "language": query.language
        }

@app.post("/upload-training-data")
async def upload_training_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global company_name
    try:
        logger.info(f"Receiving file upload: {file.filename}")
        os.makedirs("temp_data", exist_ok=True)
        file_path = f"temp_data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text_content = await process_file_content(file_path, file.filename)
        file_contents[file.filename] = text_content
        background_tasks.add_task(context_manager.update_vectors, file.filename, text_content)
        company_name = extract_company_name(text_content) or company_name
        return {"message": "File uploaded and processed successfully", "filename": file.filename, "company_name": company_name}
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_file_content(file_path: str, file_name: str) -> str:
    if file_name.endswith(".csv"):
        with open(file_path, "r") as f:
            return f.read()
    elif file_name.endswith(".txt"):
        with open(file_path, "r") as f:
            return f.read()
    elif file_name.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            return "\n".join(page.get_text() for page in doc)
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "chat":
                query_text = data["text"]
                language = data.get("language", detect_language(query_text))
                
                logger.info(f"WebSocket MTN AI request: {query_text}")
                
                # Search MTN knowledge base for relevant information
                mtn_context = search_mtn_knowledge(query_text)
                
                # Enhanced prompt for streaming with intent analysis
                intent = analyze_query_intent(query_text)
                prompt = format_mtn_prompt(query_text, language, mtn_context, "")
                
                await generate_gemini_response_stream(prompt, websocket)
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "content": f"Erreur: {str(e)}"
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