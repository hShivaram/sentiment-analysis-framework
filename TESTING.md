# Testing in Sentiment Analysis Framework

This document explains the different types of tests in the framework and their purposes.

## Test Types

### 1. Unit Tests (`test_sentiment_model.py`)

**Purpose**: Verify the basic functionality of individual components.

**Key Characteristics**:
- Tests individual methods in isolation
- Uses mocks to isolate dependencies
- Fast execution
- Runs on every commit/pull request

**Example Tests**:
- Test model initialization
- Test input validation
- Test error handling for invalid inputs
- Test output format

**When to Run**:
- During development
- Before committing code
- In CI pipeline for every push/PR

### 2. Integration Tests (`test_sentiment_model_deepeval.py`)

**Purpose**: Validate model performance and behavior with real dependencies.

**Key Characteristics**:
- Tests the model with actual dependencies
- Measures performance metrics (accuracy, precision, recall, etc.)
- Slower than unit tests
- Uses the `deepeval` library

**Example Tests**:
- Test model accuracy on test dataset
- Validate prediction confidence scores
- Test model behavior with edge cases

**When to Run**:
- Before merging to main
- After significant model changes
- In scheduled CI jobs

### 3. Model Validation Tests (`test_model_with_deepchecks.py`)

**Purpose**: Monitor model health and data quality in production.

**Key Characteristics**:
- Focuses on data and model health
- Detects data drift and anomalies
- Uses the `deepchecks` library
- More comprehensive than unit/integration tests

**Example Tests**:
- `test_data_validation`: Validates data quality and integrity
- `test_model_performance`: Checks model predictions against various metrics
- `test_data_drift`: Detects drift between training and production data

**When to Run**:
- Continuously in production
- After model deployment
- During scheduled monitoring

## Test Execution

### Running Tests Locally

```bash
# Run all tests
pytest

# Run specific test type
pytest tests/test_sentiment_model.py -v           # Unit tests
pytest tests/test_sentiment_model_deepeval.py -v  # Integration tests
pytest tests/test_model_with_deepchecks.py -v     # Model validation tests

# Run with coverage report
pytest --cov=src --cov-report=term-missing
```

### CI/CD Integration

The CI pipeline runs:
1. Unit tests on every push/PR
2. Integration tests on a schedule or manual trigger
3. Model validation tests in production

## Test Data

- Unit tests use small, controlled test cases
- Integration tests use a representative sample of real data
- Model validation tests use production-like data

## Adding New Tests

1. **Unit Tests**: Add to `test_sentiment_model.py`
2. **Integration Tests**: Add to `test_sentiment_model_deepeval.py`
3. **Model Validation**: Add to `test_model_with_deepchecks.py`

Follow existing patterns and include clear docstrings explaining the test purpose.
