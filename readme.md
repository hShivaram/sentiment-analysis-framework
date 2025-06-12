# Sentiment Analysis Framework

A robust framework for sentiment analysis using Hugging Face's Transformers library, featuring comprehensive testing and CI/CD integration.

## Features

- 🚀 Fast and accurate sentiment analysis using pre-trained models
- 🧪 Comprehensive test suite with unit, integration, and model validation tests
- 🔍 DeepChecks integration for model monitoring and validation
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
│   ├── test_sentiment_model_deepeval.py  # Integration tests
│   └── test_model_with_deepchecks.py # Model validation tests
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

### Quick Start

Run all tests:
```bash
pytest
```

### Test Types

1. **Unit Tests** (Fast, basic functionality):
   ```bash
   pytest tests/test_sentiment_model.py -v
   ```

2. **Integration Tests** (Comprehensive, with real dependencies):
   ```bash
   pytest tests/test_sentiment_model_deepeval.py -v
   ```

3. **Model Validation** (Data quality and model health):
   ```bash
   pytest tests/test_model_with_deepchecks.py -v
   ```

### Coverage Report

```bash
pytest --cov=src --cov-report=term-missing
```

For detailed information about the different test types and when to use them, see [TESTING.md](TESTING.md).

## CI/CD Pipeline

The project includes a GitHub Actions workflow that runs:
- Unit tests on every push and pull request
- Integration tests on a daily schedule
- Model validation tests in production
- Code coverage reporting

See [TESTING.md](TESTING.md) for detailed information about the test types and when they run.

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
 
