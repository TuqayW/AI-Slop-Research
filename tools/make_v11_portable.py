import joblib

SOURCE = "hc3_v11_combined_model.joblib"
TARGET = "hc3_v11_portable.joblib"


# TODO: replace this with the EXACT normalize_text function used when the
# model was trained. It only needs to exist here so joblib can unpickle
# the original bundle — it is NOT saved into the portable output.
def normalize_text(text):
    return text.strip().lower()


bundle = joblib.load(SOURCE)

portable = {
    "word_vectorizer": bundle["word_vectorizer"],
    "char_vectorizer": bundle["char_vectorizer"],
    "model": bundle["model"],
    "threshold": float(bundle["threshold"]),
}

joblib.dump(portable, TARGET, compress=3)

print("Created:", TARGET)
print("Threshold:", portable["threshold"])