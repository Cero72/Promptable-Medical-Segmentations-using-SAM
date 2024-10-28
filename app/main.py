import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import segmentation
import torch

# Initialize FastAPI app
app = FastAPI(title="Medical Image Segmentation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure torch
torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Include routers
app.include_router(segmentation.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # You can add initialization code here
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
