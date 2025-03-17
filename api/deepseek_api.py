from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from deepseek.core import Config, SecurityProvider, StorageProvider
from deepseek.ai import AIEngine

app = FastAPI(title="DeepSeek API", version="3.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize providers
security = SecurityProvider()
storage = StorageProvider()
ai_engine = AIEngine()

# Models
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    result: Any
    metadata: Dict[str, Any]

# Authentication dependency
async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        token = authorization.replace("Bearer ", "")
        payload = security.validate_token(token)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

# Routes
@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, user=Depends(get_current_user)):
    """
    Process a query through the DeepSeek AI engine.
    """
    try:
        # Check permissions
        if not security.check_permission(user, "query"):
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Process query
        result = ai_engine.process_query(
            query=request.query,
            context=request.context or {},
            options=request.options or {},
            user_id=user.get("sub")
        )
        
        # Store query history
        storage.write(
            f"history:{user.get('sub')}:{time.time()}",
            {
                "query": request.query,
                "result": result,
                "timestamp": time.time()
            },
            namespace="history"
        )
        
        return QueryResponse(
            result=result,
            metadata={
                "timestamp": time.time(),
                "model": ai_engine.current_model
            }
        )
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
