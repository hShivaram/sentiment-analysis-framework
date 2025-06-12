import pytest
from unittest.mock import patch, MagicMock
from src.sentiment_model import SentimentModel

@pytest.fixture
def mock_pipeline():
    with patch('transformers.pipeline') as mock_pipe:
        mock_pipe.return_value = MagicMock()
        yield mock_pipe.return_value

@pytest.fixture
def sentiment_model(mock_pipeline):
    # Configure the mock pipeline to return different results based on input
    def mock_pipeline_func(texts):
        results = []
        for text in texts:
            text_lower = text.lower()
            if "love" in text_lower:
                results.append({"label": "POSITIVE", "score": 0.99})
            elif "worst" in text_lower:
                results.append({"label": "NEGATIVE", "score": 0.99})
            else:
                results.append({"label": "POSITIVE", "score": 0.5})
        return results
    
    mock_pipeline.side_effect = mock_pipeline_func
    return SentimentModel()

@pytest.mark.parametrize("text,expected_label", [
    ("I love this product", "POSITIVE"),
    ("This is the worst experience ever.", "NEGATIVE"),
])
def test_analyze_known_sentiment(sentiment_model, text, expected_label, capsys):
    print(f"\nTesting text: {text}")
    result = sentiment_model.analyze(text)
    print(f"Result: {result}")
    
    # Print debug info
    with capsys.disabled():
        print("\nDebug Info:")
        print(f"- Input text: {text}")
        print(f"- Expected label: {expected_label}")
        print(f"- Actual label: {result['label']}")
        print(f"- Score: {result['score']}")
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "label" in result, "Result should contain 'label' key"
    assert "score" in result, "Result should contain 'score' key"
    assert result["label"] == expected_label, f"Expected label '{expected_label}' but got '{result['label']}'"
    assert 0.0 <= result["score"] <= 1.0, f"Score {result['score']} is not between 0 and 1"

def test_analyze_raises_on_empty_text(sentiment_model):
    with pytest.raises(ValueError, match="non-empty string"):
        sentiment_model.analyze("")

def test_analyze_raises_on_non_string(sentiment_model):
    with pytest.raises(ValueError, match="non-empty string"):
        sentiment_model.analyze(1234)