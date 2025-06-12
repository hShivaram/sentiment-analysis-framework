# src/sentiment_model.py

import logging
import time
from typing import Dict, Any
from transformers import pipeline

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add a counter to track model instances
_model_instance_count = 0

class SentimentModel:
    """Wrapper around Hugging Face sentiment-analysis pipeline."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the sentiment analysis model.
        
        Args:
            model_name: Name of the pre-trained model to use for sentiment analysis.
        """
        global _model_instance_count
        _model_instance_count += 1
        self.instance_id = _model_instance_count
        
        start_time = time.time()
        logger.info(f"[{self.instance_id}] Initializing SentimentModel with {model_name}")
        
        self.model_name = model_name
        logger.debug(f"[{self.instance_id}] Loading model pipeline...")
        
        try:
            self.pipeline = pipeline("sentiment-analysis", model=model_name)
            load_time = time.time() - start_time
            logger.info(f"[{self.instance_id}] Successfully loaded model: {model_name} in {load_time:.2f}s")
            logger.debug(f"[{self.instance_id}] Model instance details: {self}")
        except Exception as e:
            logger.error(f"[{self.instance_id}] Failed to load model: {str(e)}")
            raise

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Contains 'label' (POSITIVE/NEGATIVE) and 'score' (confidence between 0 and 1)
            
        Raises:
            ValueError: If input is empty or not a string
        """
        if not text or not isinstance(text, str):
            error_msg = "Input text must be a non-empty string."
            logger.error(f"[{getattr(self, 'instance_id', '?')}] {error_msg}")
            raise ValueError(error_msg)
        
        # Truncate text for logging if too long
        text_preview = text[:50] + ('...' if len(text) > 50 else '')
        logger.debug(f"[{self.instance_id}] Analyzing text: '{text_preview}'")
        
        try:
            start_time = time.time()
            result = self.pipeline(text)[0]
            process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            logger.debug(
                f"[{self.instance_id}] Analysis completed in {process_time:.2f}ms - "
                f"Label: {result['label']}, Score: {result['score']:.4f}"
            )
            
            return {"label": result["label"], "score": result["score"]}
            
        except Exception as e:
            logger.error(f"[{self.instance_id}] Error during analysis: {str(e)}")
            raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sentiment_model.py \"your input text here\"")
        sys.exit(1)

    text = sys.argv[1]
    model = SentimentModel()
    output = model.analyze(text)
    print(f"Sentiment: {output['label']} (Confidence: {output['score']:.2f})")
