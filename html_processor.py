# Minimal enhancement to existing html_processor.py
# Only add field_type detection, keep everything else the same

"""
Enhanced HTML processing with minimal field-type detection
"""
import concurrent.futures
import re
import json
from typing import List

import requests
import spacy
import torch
from bs4 import BeautifulSoup
from transformers import T5Tokenizer, T5ForConditionalGeneration

from config import (
    build_html_url, DOWNLOAD_TIMEOUT, MAX_CONCURRENT_DOWNLOADS,
    MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, CHUNK_OVERLAP
)
from logging_config import get_logger
from models import ArticleData, ProcessedDocument, ArticleProcessResult

logger = get_logger(__name__)

# Keep all existing classes unchanged, just enhance HTMLParser
class HTMLDownloader:
    """Class for downloading HTML content with session management"""
    # ... EXACTLY the same as your original code

    def __init__(self):
        self.session = None
        self.init_session()

    def init_session(self):
        """Initialize optimized session for HTML downloads"""
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3,
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update({
            'User-Agent': 'Medical-Article-Processor/1.0'
        })
        logger.info("SUCCESS: Download session initialized with connection pooling")

    def download_html_content(self, article_code: str) -> str:
        """Download HTML content for a specific article"""
        html_url = build_html_url(article_code)
        try:
            response = self.session.get(html_url, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            return response.text
        except requests.RequestException as e:
            logger.error(f"ERROR: Failed to download {html_url}: {e}")
            raise Exception(f"Download failed: {str(e)}")

class HTMLParser:
    """Enhanced parser with minimal field-type detection"""

    def __init__(self):
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.setup_medspacy_pipeline()

    def setup_medspacy_pipeline(self):
        """Setup pipeline - fallback to pure spaCy if MedSpaCy causes issues"""
        try:
            # Try MedSpaCy first (minimal setup)
            base_nlp = spacy.load("en_core_sci_sm")
            # Commented out for stability as in your original
            # self.nlp = medspacy.load(model=base_nlp, enable=[], disable=["medspacy_pyrush", "medspacy_context"])
            self.nlp = base_nlp
            logger.info("SUCCESS: SpaCy pipeline initialized")

        except Exception as e:
            logger.warning(f"WARNING: SpaCy failed ({e}), using basic NLP")
            self.nlp = spacy.load("en_core_sci_sm")
            logger.info("SUCCESS: Pure spaCy pipeline initialized")

    def table_to_text(self, table_html: str) -> str:
        """Convert HTML table to text using T5"""
        # EXACTLY the same as your original
        input_text = f"summarize table: {table_html}"
        inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.t5_model.generate(**inputs)
        return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def enhance_text_with_medspacy(self, text: str) -> dict:
        """Extract medical entities using MedSpaCy's built-in capabilities"""
        # EXACTLY the same as your original code
        doc = self.nlp(text)

        entities = []
        entity_labels = {}

        for ent in doc.ents:
            # Get context information if available
            context_info = {}
            if hasattr(ent._, 'is_negated'):
                context_info['is_negated'] = ent._.is_negated
            if hasattr(ent._, 'is_uncertain'):
                context_info['is_uncertain'] = ent._.is_uncertain
            if hasattr(ent._, 'is_historical'):
                context_info['is_historical'] = ent._.is_historical
            if hasattr(ent._, 'is_family'):
                context_info['is_family'] = ent._.is_family

            entity_info = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, 'score', 1.0)
            }

            # Add context info if available
            if context_info:
                entity_info["context"] = context_info

            entities.append(entity_info)

            # Count entity labels
            entity_labels[ent.label_] = entity_labels.get(ent.label_, 0) + 1

        # Extract unique medical concepts
        medical_concepts = []
        seen_concepts = set()

        for entity in entities:
            concept_key = f"{entity['text'].lower()}_{entity['label']}"
            if concept_key not in seen_concepts:
                concept_info = {
                    "concept": entity['text'].lower(),
                    "type": entity['label']
                }
                if "context" in entity:
                    concept_info["context"] = entity["context"]

                medical_concepts.append(concept_info)
                seen_concepts.add(concept_key)

        return {
            "entities": entities,
            "medical_concepts": medical_concepts,
            "entity_labels": entity_labels,
            "total_entities": len(entities),
            "unique_concepts": len(medical_concepts)
        }

    def detect_element_type(self, element) -> str:
        """Simple field type detection based on HTML element"""
        if element.name == 'h1':
            return 'title'
        elif element.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
            return 'outline'  # headings become outline
        elif element.name == 'table':
            return 'content'  # tables are content
        elif element.name == 'figcaption' or 'caption' in str(element.get('class', [])).lower():
            return 'content'  # captions are content
        else:
            return 'content'  # everything else is content

    def extract_content_from_html(self, html_content: str) -> List[dict]:
        """Extract meaningful content from HTML with minimal field-type enhancement"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove unwanted sections - EXACTLY as your original
            for elem in soup.find_all(class_=['c-ckc-bibliography', 'c-ckc-further-reading']):
                elem.decompose()
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            chunks = []
            current_chunk = []
            current_length = 0
            current_field_type = 'content'  # Default field type

            # Extract headings, paragraphs, tables, and captions - SAME logic as original
            for element in soup.find_all(
                    ['h1', 'h2', 'h3', 'p', 'table', 'figcaption'] + [{'name': 'p', 'class': 'c-ckc-figure__caption'}]):

                # Detect field type for this element
                element_field_type = self.detect_element_type(element)

                if element.name == 'table':
                    text = self.table_to_text(str(element))
                else:
                    text = element.get_text(strip=True)

                if len(text) < MIN_CHUNK_SIZE:
                    continue

                # SAME chunking logic as original, just track field type
                if current_length + len(text) > MAX_CHUNK_SIZE and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": self.enhance_text_with_medspacy(chunk_text),
                        "field_type": current_field_type  # ONLY addition
                    })
                    overlap_length = int(len(current_chunk[-1]) * CHUNK_OVERLAP)
                    current_chunk = [current_chunk[-1][-overlap_length:]] if current_chunk else []
                    current_length = sum(len(t) for t in current_chunk)

                current_chunk.append(text)
                current_length += len(text)

                # Update field type - prioritize title > outline > content
                if element_field_type == 'title':
                    current_field_type = 'title'
                elif element_field_type == 'outline' and current_field_type == 'content':
                    current_field_type = 'outline'

            # Handle final chunk - SAME as original
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": self.enhance_text_with_medspacy(chunk_text),
                    "field_type": current_field_type  # ONLY addition
                })

            return chunks

        except Exception as e:
            logger.error(f"ERROR: Failed to parse HTML: {e}")
            raise Exception(f"HTML parsing failed: {str(e)}")

class TextProcessor:
    """Class for text cleaning and processing - EXACTLY the same"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\/\%\Ã‚Â°\+\=]', '', text)
        if len(text) > MAX_CHUNK_SIZE:
            text = text[:MAX_CHUNK_SIZE].rsplit(' ', 1)[0] + '...'
        return text.strip()

    @staticmethod
    def chunk_text_intelligently(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """Split text into intelligent chunks with overlap"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        overlap_ratio = CHUNK_OVERLAP

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_length = int(len(current_chunk) * overlap_ratio)
                current_chunk = current_chunk[-overlap_length:]
            current_chunk += (" " + sentence if current_chunk else sentence)

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

class ArticleProcessor:
    """Main class for processing articles - minimal enhancement"""

    def __init__(self):
        self.downloader = HTMLDownloader()
        self.parser = HTMLParser()
        self.text_processor = TextProcessor()

    def process_article_to_documents(self, article: ArticleData) -> List[ProcessedDocument]:
        """Process a single article - SAME logic with minimal field_type addition"""
        try:
            logger.info(f"DOWNLOAD: Fetching HTML for article {article.code}")
            html_content = self.downloader.download_html_content(article.code)

            logger.info(f"PARSE: Extracting content from HTML for {article.code}")
            raw_chunks = self.parser.extract_content_from_html(html_content)

            if not raw_chunks:
                raise Exception("No content extracted from HTML")

            documents = []
            chunk_index = 0

            for chunk in raw_chunks:
                cleaned_text = self.text_processor.clean_text(chunk["text"])
                if len(cleaned_text) < MIN_CHUNK_SIZE:
                    continue

                sub_chunks = self.text_processor.chunk_text_intelligently(cleaned_text)

                for sub_chunk in sub_chunks:
                    if len(sub_chunk) < MIN_CHUNK_SIZE:
                        continue

                    # SAME metadata as original, just add field_type
                    metadata = {
                        "article_id": str(article.id),
                        "article_code": article.code,
                        "article_title": article.title,
                        "book_title": article.book_title,
                        "author": article.author,
                        "type": article.type,
                        "chunk_index": chunk_index,
                        "created_at": article.created_at,
                        "updated_at": article.updated_at,
                        "chunk_length": len(sub_chunk),
                        "source": "html_extraction",
                        # Convert complex objects to JSON strings for ChromaDB compatibility
                        "medical_entities_json": json.dumps(chunk["metadata"]["entities"]),
                        "medical_concepts_json": json.dumps(chunk["metadata"]["medical_concepts"]),
                        "entity_labels_json": json.dumps(chunk["metadata"]["entity_labels"]),
                        "total_entities": chunk["metadata"]["total_entities"],
                        "unique_concepts": chunk["metadata"]["unique_concepts"],
                        # ONLY new field
                        "field_type": chunk.get("field_type", "content")
                    }
                    if article.root_code:
                        metadata["root_code"] = article.root_code

                    document = ProcessedDocument(
                        id=f"{article.code}_chunk_{chunk_index}",
                        text=sub_chunk,
                        metadata=metadata
                    )
                    documents.append(document)
                    chunk_index += 1

            logger.info(f"SUCCESS: Processed {article.code} -> {len(documents)} chunks with field types")
            return documents
        except Exception as e:
            logger.error(f"ERROR: Failed to process article {article.code}: {e}")
            raise

    def process_multiple_articles_concurrent(self, articles: List[ArticleData]) -> tuple[
        List[ProcessedDocument], List[ArticleProcessResult]]:
        """Process multiple articles concurrently - EXACTLY the same as original"""
        all_documents = []
        results = []

        def process_single_article(article):
            try:
                documents = self.process_article_to_documents(article)
                return {
                    'success': True,
                    'article_code': article.code,
                    'documents': documents,
                    'chunk_count': len(documents)
                }
            except Exception as e:
                return {
                    'success': False,
                    'article_code': article.code,
                    'documents': [],
                    'chunk_count': 0,
                    'error': str(e)
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
            future_to_article = {
                executor.submit(process_single_article, article): article
                for article in articles
            }

            for future in concurrent.futures.as_completed(future_to_article):
                result = future.result()
                all_documents.extend(result['documents'])
                article_result = ArticleProcessResult(
                    article_code=result['article_code'],
                    success=result['success'],
                    chunk_count=result['chunk_count'],
                    error=result.get('error', '')
                )
                results.append(article_result)

        return all_documents, results