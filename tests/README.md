# Sentiment Analysis Test Suite

This directory contains test files for validating the sentiment analysis functionality in the project. There are two main test files, each serving different purposes in the testing strategy.

## Test Files Comparison

### 1. `test_sentiment_model.py`
- **Type**: Unit tests with mocks
- **Purpose**: Fast, isolated testing of the SentimentModel interface
- **Characteristics**:
  - Mocks the transformers pipeline
  - Tests error conditions and edge cases
  - Fast execution
  - No external dependencies (beyond pytest)
- **Best for**:
  - Development workflow
  - CI/CD pipelines
  - Testing the model wrapper's interface

### 2. `test_sentiment_model_deepeval.py`
- **Type**: Integration tests with real model
- **Purpose**: Comprehensive validation of model behavior
- **Characteristics**:
  - Uses the actual model for testing
  - Includes performance measurements
  - More thorough test coverage
  - Slower execution
- **Best for**:
  - Pre-release validation
  - Model behavior testing
  - Performance benchmarking

## Test File: test_sentiment_model_deepeval.py

### Overview
This test suite validates the functionality of the sentiment analysis model, ensuring it correctly identifies and classifies sentiment in text inputs. The tests cover various scenarios including positive/negative sentiment detection, consistency, edge cases, and error handling.

### Test Structure

#### 1. Positive Sentiment Testing
- **Function**: `test_positive_sentiment(sentiment_model)`
- **Purpose**: Verifies that positive sentiment phrases are correctly identified.
- **Test Cases**:
  - "I love this product!"
  - "This is absolutely amazing!"
  - "Great job, I'm very happy with the results."
  - "Excellent service, will definitely come back!"
  - "The quality is outstanding and the price is fair."
- **Assertions**:
  - Output label is "POSITIVE"
  - Confidence score is between 0.5 and 1.0

#### 2. Negative Sentiment Testing
- **Function**: `test_negative_sentiment(sentiment_model)`
- **Purpose**: Ensures negative sentiment phrases are correctly identified.
- **Test Cases**:
  - "I hate this product!"
  - "This is absolutely terrible!"
  - "Poor quality, I'm very disappointed."
  - "Worst service I've ever experienced."
  - "The product broke after one day of use."
- **Assertions**:
  - Output label is "NEGATIVE"
  - Confidence score is between 0.5 and 1.0

#### 3. Sentiment Consistency
- **Function**: `test_sentiment_consistency(sentiment_model)`
- **Purpose**: Verifies that similar phrases produce consistent sentiment outputs.
- **Test Cases**:
  - "This product is amazing!"
  - "I'm really impressed with this product!"
  - "What a fantastic product!"
- **Assertion**: All similar positive phrases should have the same sentiment label.

#### 4. Edge Cases
- **Function**: `test_edge_cases(sentiment_model)`
- **Purpose**: Tests the model with various edge cases.
- **Test Cases**:
  - Very short input ("Great!")
  - Very long input (repeated sentence)
  - Neutral statement ("This is a test.")
- **Assertion**: The model should handle these cases without errors.

#### 5. Error Handling
- **Function**: `test_error_handling(sentiment_model)`
- **Purpose**: Validates proper error handling for invalid inputs.
- **Test Cases**:
  - Empty string ("")
  - `None` value
  - Non-string input (integer)
- **Assertion**: The model should raise a `ValueError` for these cases.

### Running the Tests

To run the test suite, use the following command from the project root directory:

```bash
pytest tests/test_sentiment_model_deepeval.py -v
```

The `-v` flag enables verbose output, showing detailed test results.

### Test Fixture

The test suite uses a single instance of the `SentimentModel` for all tests, created using the `@pytest.fixture(scope="module")` decorator. This improves test performance by avoiding redundant model loading.

### Expected Output

When all tests pass, you'll see output similar to:

```
======================================== test session starts ========================================
...
test_sentiment_model_deepeval.py::test_positive_sentiment PASSED                              [ 20%]
test_sentiment_model_deepeval.py::test_negative_sentiment PASSED                              [ 40%]
test_sentiment_model_deepeval.py::test_sentiment_consistency PASSED                           [ 60%]
test_sentiment_model_deepeval.py::test_edge_cases PASSED                                       [ 80%]
test_sentiment_model_deepeval.py::test_error_handling PASSED                                  [100%]

========================================= 5 passed in X.XXs =========================================
```

## Continuous Integration (CI)

The project includes a GitHub Actions workflow that runs both test suites in different jobs:

1. **Unit Tests**: Fast tests using `test_sentiment_model.py`
   - Runs on every push and pull request
   - Quick feedback on code changes

2. **Integration Tests**: Comprehensive tests using `test_sentiment_model_deepeval.py`
   - Runs on a schedule (daily) and on release
   - More thorough validation

### Running Tests Locally

Run unit tests (fast):
```bash
pytest tests/test_sentiment_model.py -v
```

Run integration tests (comprehensive):
```bash
pytest tests/test_sentiment_model_deepeval.py -v
```

Run all tests:
```bash
pytest tests/ -v
```

### Dependencies

- Python 3.6+
- pytest
- The project's sentiment analysis model and its dependencies
- For CI:
  - GitHub Actions
  - Caching of model weights (handled automatically)

### Note

These tests are designed to be fast and reliable, with no external API dependencies. They provide comprehensive coverage of the sentiment analysis functionality while maintaining simplicity and maintainability.
