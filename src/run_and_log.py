import pandas as pd
import sys
from src.sentiment_model import SentimentModel

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_and_log.py input.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = "tests/data/model_output.csv"

    df = pd.read_csv(input_path)
    model = SentimentModel()

    outputs = []
    for text in df["text"]:
        result = model.analyze(text)
        outputs.append({"text": text, "label": result["label"], "score": result["score"]})

    pd.DataFrame(outputs).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")