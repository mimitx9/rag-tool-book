"""
Logging configuration for the embedding service
"""
import logging
import sys
import codecs


class UTF8FileHandler(logging.FileHandler):
    """Custom file handler for UTF-8 encoding"""
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)


class SafeStreamHandler(logging.StreamHandler):
    """Safe stream handler that handles encoding errors"""
    def __init__(self, stream=None):
        super().__init__(stream)

    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            msg = self.format(record)
            safe_msg = msg.encode('ascii', 'ignore').decode('ascii')
            self.stream.write(safe_msg + self.terminator)
            self.flush()


def setup_logging():
    """Setup logging configuration"""
    # Windows UTF-8 support
    if sys.platform.startswith('win'):
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
        except:
            pass

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            UTF8FileHandler('embedding_chroma_service.log'),
            SafeStreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def get_logger(name: str = __name__) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)