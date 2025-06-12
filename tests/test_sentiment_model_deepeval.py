import pytest
import logging
import sys
from src.sentiment_model import SentimentModel, logger as model_logger

# Configure logging to show debug output for our tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Get the root logger
logger = logging.getLogger()

# Create a test-specific logger
test_logger = logging.getLogger('test_sentiment_model')

# Test cases with expected sentiment labels
TEST_CASES = [
    {"input": "I love this product!", "expected": "POSITIVE"},
    {"input": "This is the worst experience ever.", "expected": "NEGATIVE"},
    {"input": "The movie was okay, not great but not terrible.", "expected": "POSITIVE"},
    {"input": "Absolutely fantastic service, highly recommended!", "expected": "POSITIVE"},
    {"input": "I'm extremely disappointed with the quality.", "expected": "NEGATIVE"},
]

# Initialize the model once for all tests
@pytest.fixture(scope="module")
def sentiment_model():
    """Fixture that provides a SentimentModel instance.
    
    Uses module scope to ensure the model is loaded only once per test module.
    """
    test_logger.info("\n=== Setting up SentimentModel fixture (module scope) ===")
    model = SentimentModel()
    test_logger.info(f"=== Created model instance with ID: {model.instance_id} ===\n")
    return model


def test_model_initialization():
    """Test that multiple model instances can be created and used independently."""
    test_logger.info("\n=== Testing model initialization ===")
    
    # Create first model instance
    test_logger.info("Creating first model instance...")
    model1 = SentimentModel()
    
    # Create second model instance
    test_logger.info("Creating second model instance...")
    model2 = SentimentModel()
    
    # Verify both instances work independently
    result1 = model1.analyze("This is great!")
    result2 = model2.analyze("This is terrible!")
    
    assert result1["label"] in ["POSITIVE", "NEGATIVE"]
    assert result2["label"] in ["POSITIVE", "NEGATIVE"]
    
    test_logger.info(f"Model 1 ID: {model1.instance_id}, Result: {result1}")
    test_logger.info(f"Model 2 ID: {model2.instance_id}, Result: {result2}")
    test_logger.info("=== Model initialization test completed ===\n")

def test_positive_sentiment(sentiment_model):
    """Test that positive sentiment is correctly identified."""
    test_logger.info("\n=== Starting positive sentiment test ===")
    
    positive_phrases = [
        "I love this product!",
        "This is absolutely amazing!",
        "Great job, I'm very happy with the results.",
        "Excellent service, will definitely come back!",
        "The quality is outstanding and the price is fair."
    ]
    
    for i, phrase in enumerate(positive_phrases, 1):
        test_logger.debug(f"Testing positive phrase {i}/{len(positive_phrases)}: {phrase[:30]}...")
        result = sentiment_model.analyze(phrase)
        assert result["label"] == "POSITIVE", f"Expected POSITIVE for: {phrase}"
        assert 0.5 <= result["score"] <= 1.0, f"Confidence score out of range for: {phrase}"
    
    test_logger.info(f"✓ Successfully tested {len(positive_phrases)} positive phrases")
    test_logger.info("=== Positive sentiment test completed ===\n")

def test_negative_sentiment(sentiment_model):
    """Test that negative sentiment is correctly identified."""
    negative_phrases = [
        "I hate this product!",
        "This is absolutely terrible!",
        "Poor quality, I'm very disappointed.",
        "Worst service I've ever experienced.",
        "The product broke after one day of use."
    ]
    
    for phrase in negative_phrases:
        result = sentiment_model.analyze(phrase)
        assert result["label"] == "NEGATIVE", f"Expected NEGATIVE for: {phrase}"
        assert 0.5 <= result["score"] <= 1.0, f"Confidence score out of range for: {phrase}"

def test_sentiment_consistency(sentiment_model):
    """Test that similar inputs produce consistent sentiment outputs."""
    similar_phrases = [
        "This product is amazing!",
        "I'm really impressed with this product!",
        "What a fantastic product!"
    ]
    
    # Get predictions for all phrases
    predictions = [sentiment_model.analyze(phrase)["label"] for phrase in similar_phrases]
    
    # Check consistency (all predictions should be the same)
    assert len(set(predictions)) == 1, \
        f"Inconsistent predictions for similar phrases: {dict(zip(similar_phrases, predictions))}"

def test_edge_cases(sentiment_model):
    """Test the model with edge cases."""
    # Test with very short input
    result = sentiment_model.analyze("Great!")
    assert result["label"] in ["POSITIVE", "NEGATIVE"]
    
    # Test with very long input
    long_text = "This is a very long text. " * 20
    result = sentiment_model.analyze(long_text)
    assert result["label"] in ["POSITIVE", "NEGATIVE"]
    
    # Test with neutral statement (should still classify as positive or negative)
    result = sentiment_model.analyze("This is a test.")
    assert result["label"] in ["POSITIVE", "NEGATIVE"]

def test_error_handling(sentiment_model):
    """Test error handling for invalid inputs."""
    # Test with empty string
    with pytest.raises(ValueError):
        sentiment_model.analyze("")
    
    # Test with None
    with pytest.raises(ValueError):
        sentiment_model.analyze(None)
    
    # Test with non-string input
    with pytest.raises(ValueError):
        sentiment_model.analyze(123)
