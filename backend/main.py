from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables FIRST before importing local modules
load_dotenv()

from rag import initialize_rag, query_rag, ingest_pdf, query_pdf_rag
import tempfile
import shutil

app = FastAPI(title="Mangesh Sambare AI Portfolio Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def query_gemini(prompt: str):
    """Query Google Gemini API for text generation"""
    if not GEMINI_API_KEY:
        return None
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        return None

def query_huggingface(prompt: str, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Query Hugging Face Inference API for text generation"""
    if not HF_API_KEY:
        return None
        
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    formatted_prompt = f"<s>[INST] You are a helpful AI assistant. {prompt} [/INST]"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {"max_new_tokens": 250, "return_full_text": False}
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                return result[0]['generated_text'].strip()
        else:
            print(f"HF API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"HF Request failed: {str(e)}")
        return None
        
    return None

def query_hf_sentiment(text: str):
    """Query Hugging Face Inference API for sentiment"""
    if not HF_API_KEY:
        return None
        
    api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                best_label = result[0][0]
                return {"label": best_label["label"], "score": best_label["score"]}
    except Exception as e:
        print(f"Sentiment failed: {str(e)}")
    return None

@app.on_event("startup")
async def startup_event():
    initialize_rag()

class ChatRequest(BaseModel):
    message: str

class PredictRequest(BaseModel):
    value1: float
    value2: float

class SentimentRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    context = query_rag(req.message)
    
    prompt = f"""You are Mangesh AI, the virtual assistant for Mangesh Sambare's portfolio.
Answer the following question about Mangesh based ONLY on the provided context. Be professional, concise, and conversational.
Do not just output raw data; understand the context and answer properly.

Context from Mangesh's Text File:
{context}

Question: {req.message}
Answer:"""

    response_text = None
    
    # Priority 1: Hugging Face
    if HF_API_KEY:
        response_text = query_huggingface(prompt)
        
    # Priority 2: Gemini (If Hugging Face fails)
    if not response_text and GEMINI_API_KEY:
        response_text = query_gemini(prompt)
        
    # Priority 3: Fallback (If both APIs are blocked or fail)
    if not response_text:
        if not context.strip():
            response_text = "I couldn't find any relevant information in Mangesh's portfolio file."
        else:
            snippets = [c.strip() for c in context.split("---") if c.strip()]
            formatted_snippets = "\n\n".join([f"• {snippet}" for snippet in snippets[:2]])
            response_text = f"I am unable to reach the AI engine to summarize this for you. However, here is the exact data I found from Mangesh's portfolio file:\n\n{formatted_snippets}"
            
    return {"reply": response_text}

@app.post("/sentiment")
async def sentiment_endpoint(req: SentimentRequest):
    result = query_hf_sentiment(req.text)
    if result:
        return result
        
    text = req.text.lower()
    if any(word in text for word in ["good", "great", "awesome", "excellent", "love", "happy"]):
        return {"label": "POSITIVE", "score": 0.95}
    elif any(word in text for word in ["bad", "terrible", "awful", "hate", "sad", "angry"]):
        return {"label": "NEGATIVE", "score": 0.92}
    else:
        return {"label": "NEUTRAL", "score": 0.60}

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    prediction = (req.value1 * 2.5) + (req.value2 * 1.2) + random.uniform(-5, 5)
    return {"prediction": round(prediction, 2)}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Must be a PDF file")
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        chunks = ingest_pdf(tmp_path)
        os.unlink(tmp_path)
        
        return {"message": f"Successfully loaded {chunks} chunks from PDF!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-pdf")
async def ask_pdf(req: ChatRequest):
    context = query_pdf_rag(req.message)
    if not context.strip():
        return {"reply": "I couldn't find anything related to that in the uploaded PDF. Please upload a PDF first or try a different question."}
        
    prompt = f"""Answer the question based ONLY on the provided PDF context.

Context:
{context}

Question: {req.message}
Answer:"""

    response_text = None
    if HF_API_KEY:
        response_text = query_huggingface(prompt)
        
    if not response_text and GEMINI_API_KEY:
        response_text = query_gemini(prompt)
    
    if not response_text:
        snippets = [c.strip() for c in context.split("---") if c.strip()]
        formatted_snippets = "\n\n".join([f"• {snippet}" for snippet in snippets[:2]])
        response_text = f"The AI model failed to respond. Here are the most relevant snippets from the PDF:\n\n{formatted_snippets}"
        
    return {"reply": response_text}
