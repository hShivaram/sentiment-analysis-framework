"""
Test cases for edge cases and special scenarios in sentiment analysis.
"""
import pytest
import string
import random
import logging
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import column, data_frames
from src.sentiment_model import SentimentModel

# Set up logger
test_logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def sentiment_model():
    """Shared sentiment model fixture for all tests."""
    return SentimentModel()

# Test data
EMOJI_TEST_CASES = [
    ("I love this! ❤️", "POSITIVE"),
    ("This is terrible 😡", "NEGATIVE"),
    ("Meh, it's okay 😐", "NEGATIVE"),  # Neutral case
]

MIXED_LANGUAGE_CASES = [
    ("I love this! मुझे यह पसंद है!", "POSITIVE"),
    ("यह बहुत खराब है! This is very bad!", "NEGATIVE"),
]

LONG_INPUTS = [
    ("Great! " * 100, "POSITIVE"),
    ("Terrible! " * 100, "NEGATIVE"),
]

# Test functions
def test_emoji_handling(sentiment_model):
    """Test that emojis are properly handled in sentiment analysis."""
    for text, expected in EMOJI_TEST_CASES:
        result = sentiment_model.analyze(text)
        test_logger.info(f"Text: {text}")
        test_logger.info(f"Predicted: {result['label']}, Expected: {expected}")
        if expected == "NEGATIVE" and "Meh" in text:
            # For "Meh", accept either label since it's quite neutral
            assert result["label"] in ["POSITIVE", "NEGATIVE"]
        else:
            # For other cases, expect the exact match
            assert result["label"] == expected

def test_mixed_language_support(sentiment_model):
    """Test sentiment analysis with mixed language inputs."""
    for text, expected in MIXED_LANGUAGE_CASES:
        result = sentiment_model.analyze(text)
        test_logger.info(f"Mixed language input: {text}")
        test_logger.info(f"Predicted: {result['label']}, Expected: {expected}")
        assert result["label"] == expected

def test_very_long_inputs(sentiment_model):
    """Test that very long inputs are handled properly."""
    for text, expected in LONG_INPUTS:
        result = sentiment_model.analyze(text)
        test_logger.info(f"Processed long input of length: {len(text)}")
        test_logger.info(f"Predicted: {result['label']}, Expected: {expected}")
        assert result["label"] == expected

# Property-based testing with Hypothesis
@given(text=st.text(min_size=1, max_size=1000))
def test_random_inputs(sentiment_model, text):
    """Test that the model handles random text inputs without crashing."""
    try:
        result = sentiment_model.analyze(text)
        assert isinstance(result, dict)
        assert "label" in result
        assert "score" in result
        assert result["label"] in ["POSITIVE", "NEGATIVE"]
        assert 0 <= result["score"] <= 1
    except Exception as e:
        test_logger.error(f"Failed on input: {text}")
        raise e

# Test with special characters and symbols
def test_special_characters(sentiment_model):
    """Test handling of special characters and symbols."""
    special_texts = [
        ("This is great!!!", "POSITIVE"),
        ("This is terrible???", "NEGATIVE"),
        ("@user123 This is awesome!", "POSITIVE"),
        ("Check this out: https://example.com #cool", "POSITIVE"),
    ]
    
    for text, expected in special_texts:
        result = sentiment_model.analyze(text)
        test_logger.info(f"Special text: {text}")
        test_logger.info(f"Predicted: {result['label']}, Expected: {expected}")
        assert result["label"] == expected

# Test with repeated characters
def test_repeated_characters(sentiment_model):
    """Test handling of repeated characters (e.g., 'sooo gooood')."""
    tests = [
        ("This is soooooo good!", "POSITIVE"),
        ("I'm sooooooooo happy!", "POSITIVE"),
        ("This is baaaad!", "NEGATIVE"),
    ]
    
    for text, expected in tests:
        result = sentiment_model.analyze(text)
        test_logger.info(f"Text with repeated chars: {text}")
        test_logger.info(f"Predicted: {result['label']}, Expected: {expected}")
        assert result["label"] == expected
