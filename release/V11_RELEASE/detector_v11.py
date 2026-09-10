import re
import sys
import joblib
from scipy.sparse import hstack


MODEL_PATH = "hc3_v11_portable.joblib"


def normalize_text(text):
    text = str(text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*\*", " ", text)
    text = re.sub(r"(?<!\w)\*(?!\w)", " ", text)

    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")

    text = text.replace("—", "-")
    text = text.replace("–", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


bundle = joblib.load(
    MODEL_PATH
)

word_vectorizer = bundle["word_vectorizer"]
char_vectorizer = bundle["char_vectorizer"]
model = bundle["model"]
threshold = float(
    bundle["threshold"]
)


def detect(text):

    text = normalize_text(text)

    word_X = word_vectorizer.transform(
        [text]
    )

    char_X = char_vectorizer.transform(
        [text]
    )

    X = hstack(
        [word_X, char_X],
        format="csr",
    )

    score = float(
        model.decision_function(X)[0]
    )

    prediction = (
        "AI"
        if score >= threshold
        else "HUMAN"
    )

    return {
        "words": len(text.split()),
        "score": score,
        "threshold": threshold,
        "margin": score - threshold,
        "prediction": prediction,
    }


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python3 detector_v11.py FILE.txt"
        )

        sys.exit(1)

    path = sys.argv[1]

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            text = f.read()

    except Exception as e:

        print(
            "ERROR:",
            e
        )

        sys.exit(1)

    if not text.strip():

        print(
            "ERROR: file is empty"
        )

        sys.exit(1)

    result = detect(text)

    print(
        "File:",
        path
    )

    print(
        "Words:",
        result["words"]
    )

    print(
        "Score:",
        round(
            result["score"],
            6
        )
    )

    print(
        "Threshold:",
        round(
            result["threshold"],
            6
        )
    )

    print(
        "Margin:",
        round(
            result["margin"],
            6
        )
    )

    print(
        "Prediction:",
        result["prediction"]
    )


if __name__ == "__main__":
    main()
