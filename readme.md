# Sentiment Analysis Framework

A robust framework for sentiment analysis using Hugging Face's Transformers library, featuring comprehensive testing and CI/CD integration.

## Features

- 🚀 Fast and accurate sentiment analysis using pre-trained models
- 🧪 Comprehensive test suite with both unit and integration tests
- 🔄 CI/CD pipeline with GitHub Actions
- 📊 Code coverage reporting
- 🔍 Detailed logging and error handling

## Project Structure

```
sentiment-analysis-framework/
├── .github/workflows/    # GitHub Actions workflows
│   └── ci.yml           # CI/CD pipeline configuration
├── src/                  # Source code
│   └── sentiment_model.py  # Main sentiment analysis model
├── tests/                # Test files
│   ├── test_sentiment_model.py       # Unit tests with mocks
│   └── test_sentiment_model_deepeval.py  # Integration tests
├── .gitignore           # Git ignore file
└── requirements.txt     # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hShivaram/sentiment-analysis-framework.git
   cd sentiment-analysis-framework
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Model

```python
from src.sentiment_model import SentimentModel

# Initialize the model
model = SentimentModel()

# Analyze text sentiment
result = model.analyze("I love this product!")
print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}")
```

### Running Tests

Run unit tests (fast):
```bash
pytest tests/test_sentiment_model.py -v
```

Run integration tests (comprehensive):
```bash
pytest tests/test_sentiment_model_deepeval.py -v
```

Run all tests with coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that runs:
- Unit tests on every push and pull request
- Integration tests on a daily schedule
- Code coverage reporting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTest](https://docs.pytest.org/)
- [GitHub Actions
│   └── notebooks/
│       └── profile_and_docs.ipynb         # Profiling & auto-generating Data Docs
└── .github/
    └── workflows/
        └── ci.yml 
 
