import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

GROQ_API_KEY = "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"

# Pydantic Model for Request
class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []

# Pydantic Model for Response
class ChatResponse(BaseModel):
    response: str
    chat_history: List[Dict[str, str]]

# Initialize FastAPI App
app = FastAPI(
    title="Shiksha AI",
    description="Educational AI Assistant API",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM and Memory
def initialize_conversation():
    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-specdec"
    )
    
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an educational assistant expert, help in notes, quiz, explaining of concepts and other things. Listen to user's query and provide solution accordingly"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    return conversation, memory

# Global conversation chain and memory
conversation_chain, conversation_memory = initialize_conversation()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Update memory with existing chat history
        conversation_memory.clear()
        for msg in request.chat_history:
            if msg['role'] == 'user':
                conversation_memory.chat_memory.add_user_message(msg['content'])
            else:
                conversation_memory.chat_memory.add_ai_message(msg['content'])
        
        # Generate response
        response = conversation_chain.invoke({"input": request.message})
        
        # Prepare updated chat history
        updated_history = request.chat_history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response['text']}
        ]
        
        return ChatResponse(
            response=response['text'], 
            chat_history=updated_history
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Additional endpoints
@app.get("/resources")
async def get_resources():
    return {
        "educational_resources": [
            {"name": "Mathematics", "link": "/math"},
            {"name": "Science", "link": "/science"},
            {"name": "Languages", "link": "/languages"}
        ]
    }

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
