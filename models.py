"""
Data models and classes for the embedding service
"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ArticleData:
    """Data class for article information"""
    id: int
    code: str
    title: str
    book_title: str
    author: str
    type: str
    root_code: str
    created_at: str
    updated_at: str


@dataclass
class ProcessedDocument:
    """Data class for processed document with metadata"""
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class ArticleProcessResult:
    """Result of article processing"""
    article_code: str
    success: bool
    chunk_count: int
    error: str = ""


class ProcessingStats:
    """Class to track processing statistics"""

    def __init__(self):
        self.total_requests = 0
        self.total_texts_processed = 0
        self.total_processing_time = 0
        self.total_articles_processed = 0
        self.total_html_downloads = 0

    def increment_request(self):
        self.total_requests += 1

    def add_texts(self, count: int):
        self.total_texts_processed += count

    def add_processing_time(self, time: float):
        self.total_processing_time += time

    def add_articles(self, count: int):
        self.total_articles_processed += count

    def increment_downloads(self):
        self.total_html_downloads += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'total_requests': self.total_requests,
            'total_texts_processed': self.total_texts_processed,
            'total_processing_time': round(self.total_processing_time, 3),
            'total_articles_processed': self.total_articles_processed,
            'total_html_downloads': self.total_html_downloads,
            'average_processing_time': round(
                self.total_processing_time / self.total_requests, 3
            ) if self.total_requests > 0 else 0,
            'average_texts_per_request': round(
                self.total_texts_processed / self.total_requests, 2
            ) if self.total_requests > 0 else 0,
            'throughput_texts_per_second': round(
                self.total_texts_processed / self.total_processing_time, 2
            ) if self.total_processing_time > 0 else 0,
            'throughput_articles_per_second': round(
                self.total_articles_processed / self.total_processing_time, 2
            ) if self.total_processing_time > 0 else 0
        }