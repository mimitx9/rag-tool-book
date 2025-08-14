"""
Configuration module for the enhanced Python service
"""
import os
import torch

# Environment setup
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Model configuration
MODEL_NAME = 'ncbi/MedCPT-Article-Encoder'
QUERY_MODEL_NAME = 'ncbi/MedCPT-Query-Encoder'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

# Performance configuration
OPTIMIZED_MAX_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 512
ENABLE_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
STREAMING_THRESHOLD = 1000
THREAD_POOL_SIZE = 8
DOWNLOAD_TIMEOUT = 30  # seconds
MAX_CONCURRENT_DOWNLOADS = 10

# HTML configuration
HTML_BASE_URL = "https://storage.googleapis.com/dl.dentistrykey.com/Document"
HTML_URL_PATTERN = "{base_url}/{code}.html"

# ChromaDB configuration
CHROMA_HOST = "https://chromadb.ap.ngrok.io"
CHROMA_PORT = 443
CHROMA_COLLECTION_NAME = "clinicalkey"

# Text processing configuration
MIN_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 400
MIN_PARAGRAPH_LENGTH = 50
MIN_HEADING_LENGTH = 10
CHUNK_OVERLAP = 0.2  # 20% overlap
FLASK_HOST = 'localhost'
FLASK_PORT = '5001'
FLASK_DEBUG = 'true'


def build_html_url(article_code: str) -> str:
    """Build URL from article code"""
    return HTML_URL_PATTERN.format(base_url=HTML_BASE_URL, code=article_code)
