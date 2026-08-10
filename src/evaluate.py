import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

MODEL_PATH = "../models/bilstm_model.keras"
TOKENIZER_PATH = "../models/tokenizer.pkl"
TEST_PATH = "../data/test.csv"

model = load_model(MODEL_PATH, compile=False)


with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


test_df = pd.read_csv(TEST_PATH)

X_test = test_df["text"]
y_test = test_df["label"]


X_test_seq = tokenizer.texts_to_sequences(X_test)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)


pred_probs = model.predict(X_test_pad)
predictions = (pred_probs > 0.5).astype(int).flatten()


print("\nEvaluation results")

print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"Recall:    {recall_score(y_test, predictions):.4f}")
print(f"F1-score:  {f1_score(y_test, predictions):.4f}")


cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Phishing", "Phishing"],
    yticklabels=["Not Phishing", "Phishing"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("../images/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


print("\nClassification report:")
print(classification_report(y_test, predictions))