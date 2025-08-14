# Simple enhancement to chromadb_service.py
# Add ONE simple hybrid search method, keep everything else the same

"""
ChromaDB service with simple hybrid search enhancement
"""
import chromadb
import spacy
from chromadb.config import Settings
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from collections import defaultdict

from config import CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION_NAME, CROSS_ENCODER_MODEL
from embedding_service import embedding_service
from models import ProcessedDocument
from logging_config import get_logger

logger = get_logger(__name__)


class ChromaDBService:
    """Service for managing ChromaDB operations with simple hybrid search"""

    def __init__(self):
        self.client = None
        self.collection = None
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    # Keep ALL existing methods exactly the same
    def init_client(self) -> bool:
        """Initialize ChromaDB client"""
        try:
            self.client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                settings=Settings(anonymized_telemetry=False)
            )

            # Always refresh collection to avoid ID issues
            self.collection = self._get_or_create_collection()

            logger.info(f"SUCCESS: ChromaDB client initialized with collection {CHROMA_COLLECTION_NAME}")
            return True
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize ChromaDB: {str(e)}")
            return False

    def _get_or_create_collection(self):
        """Get or create collection with proper error handling"""
        try:
            # Try to get existing collection first
            try:
                collection = self.client.get_collection(name=CHROMA_COLLECTION_NAME)
                logger.info(f"SUCCESS: Retrieved existing collection {CHROMA_COLLECTION_NAME}")
                return collection
            except Exception:
                # Collection doesn't exist, create it WITHOUT embedding function
                logger.info(f"Collection {CHROMA_COLLECTION_NAME} not found, creating new one")
                collection = self.client.create_collection(
                    name=CHROMA_COLLECTION_NAME,
                    metadata={"hnsw:space": "l2"},
                    embedding_function=None  # IMPORTANT: Don't use ChromaDB's auto-embedding
                )
                logger.info(f"SUCCESS: Created new collection {CHROMA_COLLECTION_NAME} without auto-embedding")
                return collection
        except Exception as e:
            logger.error(f"ERROR: Failed to get/create collection: {str(e)}")
            raise

    def store_documents(self, documents: List[ProcessedDocument], embeddings: List[List[float]]) -> bool:
        """Store documents and embeddings in ChromaDB"""
        try:
            # Refresh collection reference before storing
            self.collection = self._get_or_create_collection()

            ids = [doc.id for doc in documents]
            texts = [doc.text for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"SUCCESS: Stored {len(documents)} documents in ChromaDB")
            return True
        except Exception as e:
            logger.error(f"ERROR: Failed to store documents: {str(e)}")
            return False

    def query_collection(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query ChromaDB with semantic search and reranking - EXACTLY the same as original"""
        try:
            # Refresh collection reference before querying
            self.collection = self._get_or_create_collection()

            # Generate query embedding
            query_embedding = embedding_service.generate_embeddings([query], is_query=True)[0]

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # Get more results for reranking
                include=["documents", "metadatas", "distances"]
            )

            # Check if we have results
            if not results["ids"][0]:
                return []

            # Rerank with cross-encoder
            pairs = [(query, results["documents"][0][i]) for i in range(len(results["ids"][0]))]
            scores = self.cross_encoder.predict(pairs)
            reranked_indices = scores.argsort()[-n_results:][::-1]

            # Prepare final results
            final_results = []
            for i in reranked_indices:
                result = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "rerank_score": float(scores[i])
                }
                final_results.append(result)

            return final_results
        except Exception as e:
            logger.error(f"ERROR: Failed to query collection: {str(e)}")
            return []

    def hybrid_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        try:
            self.collection = self._get_or_create_collection()
            title_weight = 0.5  # Highest priority
            content_weight = 0.4  # Medium priority
            outline_weight = 0.1  # Lowest priority
            medical_boost = 0.2  # Bonus for medical entities
        except Exception as e:
            logger.error(f"ERROR: Failed to query collection: {str(e)}")
            return []

    def _simulate_fulltext_search(self, query: str, field_type: str, n_results: int) -> List[Dict[str, Any]]:
        """
        Simulate full-text search using embedding service
        Since we can't use ChromaDB's query_texts (dimension mismatch),
        we'll use embeddings but with different query preprocessing for more literal matching
        """
        try:
            self.collection = self._get_or_create_collection()

            # For full-text effect, we can:
            # 1. Use exact query as-is (more literal)
            # 2. Maybe enhance with keyword extraction later

            # Create embedding for literal query
            literal_embedding = embedding_service.generate_embeddings([f"exact: {query}"], is_query=True)[0]

            results = self.collection.query(
                query_embeddings=[literal_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                where={"field_type": field_type}
            )

            # Post-process to boost exact keyword matches
            formatted_results = self._format_results(results, f"fulltext_{field_type}")

            # Boost results that contain exact query words
            query_words = set(query.lower().split())
            for result in formatted_results:
                text_words = set(result['text'].lower().split())
                word_overlap = len(query_words.intersection(text_words)) / len(query_words)

                # Adjust distance based on word overlap (lower distance = better match)
                original_distance = result['distance']
                boosted_distance = original_distance * (1 - word_overlap * 0.3)  # Up to 30% boost
                result['distance'] = max(0.1, boosted_distance)  # Don't go below 0.1
                result['word_overlap_score'] = word_overlap

            return formatted_results

        except Exception as e:
            logger.error(f"ERROR: Simulated full-text search failed: {str(e)}")
            return []

            query_embedding = embedding_service.generate_embeddings([query], is_query=True)[0]

            # 1. Search in title chunks (semantic only)
            title_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,
                include=["documents", "metadatas", "distances"],
                where={"field_type": "title"}
            )

            # 2. Search in content chunks - HYBRID: semantic + full-text
            # 2a. Semantic search in content
            content_semantic_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 3,  # Get more for combining
                include=["documents", "metadatas", "distances"],
                where={"field_type": "content"}
            )

            # For content: combine semantic and full-text
            semantic_weight = 0.7  # Semantic search weight within content
            fulltext_weight = 0.3  # Full-text search weight within content

            query_embedding = embedding_service.generate_embeddings([query], is_query=True)[0]

            # 1. Search in title chunks (semantic only)
            title_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,
                include=["documents", "metadatas", "distances"],
                where={"field_type": "title"}
            )

            # 2. Search in content chunks - HYBRID: semantic + full-text-like
            # 2a. Semantic search in content
            content_semantic_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 3,  # Get more for combining
                include=["documents", "metadatas", "distances"],
                where={"field_type": "content"}
            )

            # 2b. Full-text-like search in content using our embedding service
            content_fulltext_results = self._simulate_fulltext_search(query, "content", n_results * 3)

            # 3. Search in outline chunks (semantic only)
            outline_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,
                include=["documents", "metadatas", "distances"],
                where={"field_type": "outline"}
            )

            # 3. Search in outline chunks (semantic only)
            outline_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,
                include=["documents", "metadatas", "distances"],
                where={"field_type": "outline"}
            )

            # 4. Combine results with weights
            combined_scores = defaultdict(lambda: {
                'best_distance': float('inf'),
                'weighted_score': 0,
                'result': None,
                'field_matches': [],
                'search_types': []  # Track which search types matched
            })

            # Process title results (semantic only)
            for i, doc_id in enumerate(title_results["ids"][0]):
                distance = title_results["distances"][0][i]
                similarity_score = 1 / (1 + distance)  # Convert distance to similarity
                position_decay = 1 / (1 + i * 0.1)  # Position-based decay
                weighted_score = similarity_score * position_decay * title_weight

                combined_scores[doc_id]['weighted_score'] += weighted_score
                combined_scores[doc_id]['best_distance'] = min(combined_scores[doc_id]['best_distance'], distance)
                combined_scores[doc_id]['field_matches'].append('title')
                combined_scores[doc_id]['search_types'].append('semantic')
                combined_scores[doc_id]['result'] = {
                    "id": doc_id,
                    "text": title_results["documents"][0][i],
                    "metadata": title_results["metadatas"][0][i],
                    "distance": distance
                }

            # Process content results - HYBRID combination
            content_scores = defaultdict(lambda: {
                'semantic_score': 0,
                'fulltext_score': 0,
                'best_distance': float('inf'),
                'result': None
            })

            # Process semantic content results
            for i, doc_id in enumerate(content_semantic_results["ids"][0]):
                distance = content_semantic_results["distances"][0][i]
                similarity_score = 1 / (1 + distance)
                position_decay = 1 / (1 + i * 0.1)
                content_scores[doc_id]['semantic_score'] = similarity_score * position_decay
                content_scores[doc_id]['best_distance'] = min(content_scores[doc_id]['best_distance'], distance)
                content_scores[doc_id]['result'] = {
                    "id": doc_id,
                    "text": content_semantic_results["documents"][0][i],
                    "metadata": content_semantic_results["metadatas"][0][i],
                    "distance": distance
                }

            # Process full-text content results
            for i, doc_id in enumerate(content_fulltext_results["ids"][0]):
                distance = content_fulltext_results["distances"][0][i]
                # For full-text, lower distance still means better match
                fulltext_similarity = 1 / (1 + distance)
                position_decay = 1 / (1 + i * 0.1)
                content_scores[doc_id]['fulltext_score'] = fulltext_similarity * position_decay
                content_scores[doc_id]['best_distance'] = min(content_scores[doc_id]['best_distance'], distance)
                if content_scores[doc_id]['result'] is None:
                    content_scores[doc_id]['result'] = {
                        "id": doc_id,
                        "text": content_fulltext_results["documents"][0][i],
                        "metadata": content_fulltext_results["metadatas"][0][i],
                        "distance": distance
                    }

            # Combine semantic + full-text scores for content
            for doc_id, scores in content_scores.items():
                if scores['result']:
                    # Hybrid content score = semantic * 0.7 + fulltext * 0.3
                    hybrid_content_score = (
                            scores['semantic_score'] * semantic_weight +
                            scores['fulltext_score'] * fulltext_weight
                    )
                    final_content_score = hybrid_content_score * content_weight

                    combined_scores[doc_id]['weighted_score'] += final_content_score
                    combined_scores[doc_id]['best_distance'] = min(
                        combined_scores[doc_id]['best_distance'],
                        scores['best_distance']
                    )
                    combined_scores[doc_id]['field_matches'].append('content')

                    # Track which search types contributed
                    search_types = []
                    if scores['semantic_score'] > 0:
                        search_types.append('semantic')
                    if scores['fulltext_score'] > 0:
                        search_types.append('fulltext')
                    combined_scores[doc_id]['search_types'].extend(search_types)

                    if combined_scores[doc_id]['result'] is None:
                        combined_scores[doc_id]['result'] = scores['result']

            # Process outline results (semantic only)
            for i, doc_id in enumerate(outline_results["ids"][0]):
                distance = outline_results["distances"][0][i]
                similarity_score = 1 / (1 + distance)
                position_decay = 1 / (1 + i * 0.1)
                weighted_score = similarity_score * position_decay * outline_weight

                combined_scores[doc_id]['weighted_score'] += weighted_score
                combined_scores[doc_id]['best_distance'] = min(combined_scores[doc_id]['best_distance'], distance)
                combined_scores[doc_id]['field_matches'].append('outline')
                combined_scores[doc_id]['search_types'].append('semantic')
                if combined_scores[doc_id]['result'] is None:
                    combined_scores[doc_id]['result'] = {
                        "id": doc_id,
                        "text": outline_results["documents"][0][i],
                        "metadata": outline_results["metadatas"][0][i],
                        "distance": distance
                    }

            # 5. Apply medical entity boost (use existing medical data)
            for doc_id, score_data in combined_scores.items():
                if score_data['result']:
                    # Get medical entity count from existing metadata
                    total_entities = score_data['result']['metadata'].get('total_entities', 0)
                    unique_concepts = score_data['result']['metadata'].get('unique_concepts', 0)

                    # Simple medical boost based on existing data
                    if total_entities > 0:
                        entity_boost = min(total_entities * 0.05, medical_boost)  # Cap the boost
                        concept_boost = min(unique_concepts * 0.03, medical_boost * 0.5)
                        score_data['weighted_score'] += entity_boost + concept_boost

            # 6. Sort by weighted score and prepare final results
            sorted_items = sorted(
                [(doc_id, data) for doc_id, data in combined_scores.items() if data['result']],
                key=lambda x: x[1]['weighted_score'],
                reverse=True
            )

            # 7. Apply cross-encoder reranking on top results
            top_candidates = sorted_items[:n_results * 2]
            if top_candidates:
                pairs = [(query, item[1]['result']['text']) for item in top_candidates]
                cross_scores = self.cross_encoder.predict(pairs)

                # Combine weighted score with cross-encoder score
                final_results = []
                for i, (doc_id, score_data) in enumerate(top_candidates):
                    result = score_data['result'].copy()
                    result['hybrid_score'] = score_data['weighted_score']
                    result['cross_encoder_score'] = float(cross_scores[i])
                    result['final_score'] = score_data['weighted_score'] * 0.7 + cross_scores[i] * 0.3
                    result['field_matches'] = list(set(score_data['field_matches']))  # Remove duplicates
                    result['search_types'] = list(set(score_data['search_types']))  # Remove duplicates
                    result['distance'] = score_data['best_distance']

                    # Add detailed score breakdown for debugging
                    result['score_breakdown'] = {
                        'hybrid_score': score_data['weighted_score'],
                        'cross_encoder_score': float(cross_scores[i]),
                        'final_score': result['final_score'],
                        'field_matches': result['field_matches'],
                        'search_types': result['search_types']
                    }

                    final_results.append(result)

                # Final sort by combined score
                final_results.sort(key=lambda x: x['final_score'], reverse=True)

                # Log hybrid search details
                logger.info(f"HYBRID SEARCH DETAILS for '{query}':")
                logger.info(f"  Title results: {len(title_results['ids'][0])}")
                logger.info(f"  Content semantic: {len(content_semantic_results['ids'][0])}")
                logger.info(f"  Content fulltext: {len(content_fulltext_results['ids'][0])}")
                logger.info(f"  Outline results: {len(outline_results['ids'][0])}")
                logger.info(f"  Combined unique results: {len(combined_scores)}")
                logger.info(f"  Final results after reranking: {len(final_results)}")

                return final_results[:n_results]

            return []

        except Exception as e:
            logger.error(f"ERROR: Hybrid search failed: {str(e)}")
            return []

        except Exception as e:
            logger.error(f"ERROR: Hybrid search failed: {str(e)}")
            return []

    # Keep ALL other existing methods exactly the same
    def list_collections(self) -> List[str]:
        """List all collections in ChromaDB"""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"ERROR: Failed to list collections: {str(e)}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"SUCCESS: Deleted collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"ERROR: Failed to delete collection {collection_name}: {str(e)}")
            return False

    def reset_collection(self) -> bool:
        """Reset collection by deleting and recreating it"""
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(CHROMA_COLLECTION_NAME)
                logger.info(f"Deleted existing collection {CHROMA_COLLECTION_NAME}")
            except Exception:
                pass  # Collection might not exist

            # Create new collection WITHOUT embedding function
            self.collection = self.client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "l2"},
                embedding_function=None  # IMPORTANT: Don't use ChromaDB's auto-embedding
            )
            logger.info(f"SUCCESS: Reset collection {CHROMA_COLLECTION_NAME} without auto-embedding")
            return True
        except Exception as e:
            logger.error(f"ERROR: Failed to reset collection: {str(e)}")
            return False


# Global instance
chromadb_service = ChromaDBService()