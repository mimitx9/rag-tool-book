"""
Embedding processing service using Transformers for MedCPT
"""
import time
import gc
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

from config import (
    MODEL_NAME, QUERY_MODEL_NAME, ENABLE_GPU, OPTIMIZED_MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH, STREAMING_THRESHOLD
)
from logging_config import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    """Service for generating embeddings using MedCPT"""

    def __init__(self):
        self.model = None
        self.query_model = None
        self.tokenizer = None
        self.query_tokenizer = None
        self.model_loaded = False
        self.device = 'cpu'

    def load_model(self) -> bool:
        """Load the embedding models with optimizations"""
        logger.info("=" * 60)
        logger.info("STARTING EMBEDDING SERVICE")
        logger.info("=" * 60)

        try:
            logger.info(f"Loading article model: {MODEL_NAME}")
            logger.info(f"Loading query model: {QUERY_MODEL_NAME}")
            logger.info(f"GPU/MPS Available: {ENABLE_GPU}")
            logger.info(f"Optimized batch size: {OPTIMIZED_MAX_BATCH_SIZE}")

            start_time = time.time()
            self.model = AutoModel.from_pretrained(MODEL_NAME)
            self.query_model = AutoModel.from_pretrained(QUERY_MODEL_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)

            if ENABLE_GPU:
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.model = self.model.cuda()
                    self.query_model = self.query_model.cuda()
                    logger.info("SUCCESS: Models moved to CUDA")
                elif torch.backends.mps.is_available():
                    self.device = 'mps'
                    self.model = self.model.to('mps')
                    self.query_model = self.query_model.to('mps')
                    logger.info("SUCCESS: Models moved to MPS")
                try:
                    self.model = self.model.half()
                    self.query_model = self.query_model.half()
                    logger.info("OPTIMIZATION: Using half precision (FP16)")
                except:
                    logger.warning("WARNING: Half precision not supported, using FP32")
            else:
                logger.info("INFO: Using CPU")

            self.model.eval()
            self.query_model.eval()

            # Warm up the models
            warmup_texts = ["Medical warmup text for embedding model."] * min(32, OPTIMIZED_MAX_BATCH_SIZE)
            _ = self.process_embeddings_large_batch(warmup_texts, is_query=False)
            _ = self.process_embeddings_large_batch(warmup_texts, is_query=True)

            load_time = time.time() - start_time
            logger.info(f"SUCCESS: Models loaded and warmed up in {load_time:.2f} seconds")

            self.model_loaded = True
            return True

        except Exception as e:
            logger.error(f"ERROR: Failed to load model: {str(e)}")
            return False

    def preprocess_texts_optimized(self, texts: List[str]) -> List[str]:
        """Optimized text preprocessing"""
        processed = []
        for text in texts:
            text = text.strip()
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
            if not text:
                text = "[EMPTY]"
            processed.append(text)
        return processed

    def process_embeddings_large_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if not texts or not self.model_loaded:
            return []

        processed_texts = self.preprocess_texts_optimized(texts)
        all_embeddings = []
        model = self.query_model if is_query else self.model
        tokenizer = self.query_tokenizer if is_query else self.tokenizer

        for i in range(0, len(processed_texts), OPTIMIZED_MAX_BATCH_SIZE):
            batch = processed_texts[i:i + OPTIMIZED_MAX_BATCH_SIZE]

            with torch.no_grad():
                encoded = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors='pt',
                    max_length=MAX_TEXT_LENGTH
                ).to(self.device)
                outputs = model(**encoded)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())

            if ENABLE_GPU and i % (OPTIMIZED_MAX_BATCH_SIZE * 4) == 0:
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()

        return all_embeddings

    def process_embeddings_streaming(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Process very large batches with streaming"""
        logger.info(f"STREAMING: Processing {len(texts)} texts in streaming mode")

        all_embeddings = []
        chunk_size = OPTIMIZED_MAX_BATCH_SIZE * 4

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_embeddings = self.process_embeddings_large_batch(chunk, is_query)
            all_embeddings.extend(chunk_embeddings)

            if i % (chunk_size * 2) == 0:
                gc.collect()
                if ENABLE_GPU:
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device == 'mps':
                        torch.mps.empty_cache()

            logger.info(f"STREAMING: Processed {min(i + chunk_size, len(texts))}/{len(texts)} texts")

        return all_embeddings

    def generate_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Main method to generate embeddings with automatic mode selection"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        if len(texts) > STREAMING_THRESHOLD:
            return self.process_embeddings_streaming(texts, is_query)
        else:
            return self.process_embeddings_large_batch(texts, is_query)

    def cleanup(self):
        """Cleanup GPU/MPS memory and resources"""
        if ENABLE_GPU:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            elif self.device == 'mps':
                torch.mps.empty_cache()
        gc.collect()

    def is_ready(self) -> bool:
        """Check if the service is ready"""
        return self.model_loaded and self.model is not None and self.query_model is not None

# Global instance
embedding_service = EmbeddingService()