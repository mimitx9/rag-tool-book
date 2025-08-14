"""
Main application entry point for the enhanced Python embedding service
"""
import concurrent.futures
import json

import numpy as np
from flask import Flask, request, jsonify
from typing import List

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, THREAD_POOL_SIZE, CHROMA_COLLECTION_NAME
from logging_config import setup_logging
from embedding_service import embedding_service
from chromadb_service import chromadb_service
from html_processor import ArticleProcessor
from models import ArticleData, ArticleProcessResult

# Initialize logging
logger = setup_logging()

# Initialize Flask app
app = Flask(__name__)

# Global thread pool
thread_pool = None


# Hàm chuyển đổi JSON tùy chỉnh
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# Initialize Flask app
app = Flask(__name__)
app.json_encoder = NpEncoder  # Cấu hình Flask sử dụng NpEncoder


def init_thread_pool():
    """Initialize thread pool for concurrent operations"""
    global thread_pool
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
    logger.info(f"SUCCESS: Thread pool initialized with {THREAD_POOL_SIZE} workers")


def initialize_services() -> bool:
    """Initialize all services"""
    logger.info("=" * 60)
    logger.info("STARTING ENHANCED EMBEDDING + CHROMA SERVICE")
    logger.info("=" * 60)

    # Initialize embedding service
    if not embedding_service.load_model():
        logger.error("ERROR: Failed to load embedding model")
        return False

    # Initialize ChromaDB service
    if not chromadb_service.init_client():
        logger.error("ERROR: Failed to initialize ChromaDB")
        return False

    # Initialize thread pool
    init_thread_pool()

    return True


@app.route('/chroma/collection/hybrid-query', methods=['POST'])
def hybrid_query_collection():
    """Simple hybrid query: one query searches title/content/outline with smart ranking"""
    try:
        data = request.get_json()
        query = data['query']
        n_results = data.get('n_results', 10)

        logger.info(f"HYBRID QUERY: '{query}' (n_results={n_results})")

        import time
        start_time = time.time()

        # Use the new hybrid search method
        results = chromadb_service.hybrid_search(query, n_results)

        end_time = time.time()
        search_time = end_time - start_time

        # Simple analytics
        field_distribution = {}
        search_type_distribution = {}
        medical_stats = {
            "total_entities": 0,
            "avg_entities": 0,
            "results_with_entities": 0
        }

        for result in results:
            # Count field matches
            field_matches = result.get('field_matches', [])
            for field in field_matches:
                field_distribution[field] = field_distribution.get(field, 0) + 1

            # Count search types (semantic, fulltext)
            search_types = result.get('search_types', [])
            for search_type in search_types:
                search_type_distribution[search_type] = search_type_distribution.get(search_type, 0) + 1

            # Medical stats
            entities = result['metadata'].get('total_entities', 0)
            medical_stats["total_entities"] += entities
            if entities > 0:
                medical_stats["results_with_entities"] += 1

        if results:
            medical_stats["avg_entities"] = round(medical_stats["total_entities"] / len(results), 2)

        logger.info(f"SUCCESS: Hybrid search completed in {search_time:.3f}s")
        logger.info(f"  Found {len(results)} results")
        logger.info(f"  Field distribution: {field_distribution}")
        logger.info(f"  Search type distribution: {search_type_distribution}")
        logger.info(
            f"  Medical entities: {medical_stats['total_entities']} total, {medical_stats['results_with_entities']} results with entities")

        return jsonify({
            "status": "success",
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_time": search_time,
            "analytics": {
                "field_distribution": field_distribution,
                "search_type_distribution": search_type_distribution,  # NEW
                "medical_stats": medical_stats,
                "search_type": "hybrid"
            }
        })

    except Exception as e:
        logger.error(f"ERROR: Failed to perform hybrid query: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "results": [],
            "total_found": 0
        }), 500


# Also update the log_available_endpoints function to include the new endpoint
def log_available_endpoints():
    """Log all available endpoints"""
    logger.info("ENHANCED ENDPOINTS: Available endpoints:")
    logger.info(" ARTICLE PROCESSING:")
    logger.info(" - POST /chroma/articles/process - Process articles (download HTML + embed + store)")
    logger.info(" EMBEDDING:")
    logger.info(" - POST /embed - Generate embeddings (legacy)")
    logger.info(" CHROMADB:")
    logger.info(" - POST /chroma/collection/query - Query collection (semantic only)")
    logger.info(" - POST /chroma/collection/hybrid-query - Hybrid query (title + content + outline)")  # NEW
    logger.info(" - GET /chroma/collections - List all collections")
    logger.info(" SYSTEM:")
    logger.info(" - GET /health - Health check with stats")
    logger.info(" - GET /stats - Detailed processing statistics")
    logger.info("=" * 60)


def log_enhanced_features():
    """Log enhanced features"""
    from config import MAX_CONCURRENT_DOWNLOADS, DOWNLOAD_TIMEOUT
    logger.info("ENHANCED FEATURES:")
    logger.info(" - HTML download & parsing with BeautifulSoup")
    logger.info(" - Intelligent text chunking with overlap")
    logger.info(" - UMLS integration for medical concepts")
    logger.info(" - Table processing with T5")
    logger.info(" - Concurrent article processing")
    logger.info(f" - Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")
    logger.info(f" - Download timeout: {DOWNLOAD_TIMEOUT}s")
    logger.info(" - Connection pooling enabled")
    logger.info(" - GPU/MPS optimization (if available)")
    logger.info(" - Streaming processing for large batches")
    logger.info("=" * 60)


# API routes
@app.route('/chroma/articles/process', methods=['POST'])
def process_articles():
    """Process articles: download HTML, extract content, embed, store in ChromaDB"""
    try:
        data = request.get_json()
        articles = [
            ArticleData(
                id=article['id'],
                code=article['code'],
                title=article['title'],
                book_title=article.get('book_title', ''),
                author=article.get('author', ''),
                type=article.get('type', ''),
                root_code=article.get('root_code', ''),
                created_at=article.get('created_at', ''),
                updated_at=article.get('updated_at', '')
            ) for article in data['articles']
        ]

        article_processor = ArticleProcessor()
        documents, results = article_processor.process_multiple_articles_concurrent(articles)
        embeddings = embedding_service.generate_embeddings([doc.text for doc in documents])

        chromadb_service.store_documents(documents, embeddings)

        return jsonify({
            "status": "success",
            "results": [vars(result) for result in results],
            "document_count": len(documents)
        })
    except Exception as e:
        logger.error(f"ERROR: Failed to process articles: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/chroma/collection/query', methods=['POST'])
def query_collection():
    """Query ChromaDB with a search term"""
    try:
        data = request.get_json()
        query = data['query']
        n_results = data.get('n_results', 10)

        results = chromadb_service.query_collection(query, n_results)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        logger.error(f"ERROR: Failed to query collection: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/chroma/collections', methods=['GET'])
def list_collections():
    """List all ChromaDB collections"""
    try:
        collections = chromadb_service.list_collections()
        return jsonify({
            "status": "success",
            "collections": collections
        })
    except Exception as e:
        logger.error(f"ERROR: Failed to list collections: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "embedding_service_ready": embedding_service.is_ready(),
        "chromadb_service_ready": chromadb_service.collection is not None
    })


def main():
    """Main application entry point"""
    try:
        # Initialize all services
        if not initialize_services():
            logger.error("ERROR: Failed to initialize services")
            exit(1)

        # Log available endpoints and features
        log_available_endpoints()
        log_enhanced_features()

        # Start Flask application
        logger.info(f"Starting Flask server on {FLASK_HOST}:{FLASK_PORT}")
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=FLASK_DEBUG,
            threaded=True
        )

    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"ERROR: Service failed to start: {e}")
        exit(1)
    finally:
        # Cleanup
        if thread_pool:
            thread_pool.shutdown(wait=True)
        embedding_service.cleanup()
        logger.info("Service shutdown complete")


if __name__ == '__main__':
    main()
