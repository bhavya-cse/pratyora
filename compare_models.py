import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from train_model import load_and_preprocess_dataset

REAL_THRESHOLD = 0.40
FAKE_THRESHOLD = 0.65

MODELS = {
    "Current production model": "models/fake_news_model.joblib",
    "New candidate model": "models/candidate_fake_news_model.joblib",
}


def load_pipeline(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    artifact = joblib.load(path)

    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
    else:
        pipeline = artifact

    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(f"{path} does not support predict_proba().")

    classifier = pipeline.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "classes_"):
        raise TypeError(f"Could not identify classifier classes in {path}.")

    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(f"Could not find the fake-news class in {path}.")

    return pipeline, classes.index(1)


def evaluate(name, path, texts, actual):
    pipeline, fake_index = load_pipeline(path)
    probabilities = pipeline.predict_proba(texts)[:, fake_index]

    # Website-style predictions: 0=Real, 1=Fake, -1=Inconclusive
    predicted = np.full(len(actual), -1, dtype=int)
    predicted[probabilities <= REAL_THRESHOLD] = 0
    predicted[probabilities >= FAKE_THRESHOLD] = 1

    real_mask = actual == 0
    fake_mask = actual == 1
    decisive = predicted != -1

    false_fake = int(np.sum(real_mask & (predicted == 1)))
    false_real = int(np.sum(fake_mask & (predicted == 0)))
    true_real = int(np.sum(real_mask & (predicted == 0)))
    true_fake = int(np.sum(fake_mask & (predicted == 1)))
    inconclusive = int(np.sum(~decisive))

    # Standard 50% cutoff, reported separately for comparison.
    standard_predicted = (probabilities >= 0.50).astype(int)
    standard_accuracy = accuracy_score(actual, standard_predicted)
    standard_f1 = f1_score(
        actual, standard_predicted, average="macro", zero_division=0
    )

    decisive_accuracy = (
        (true_real + true_fake) / int(np.sum(decisive))
        if np.sum(decisive) else 0.0
    )

    print(f"\n{'=' * 58}")
    print(name)
    print(f"File: {path}")
    print(f"{'=' * 58}")
    print(f"Test examples:                    {len(actual)}")
    print(f"Actual real examples:             {int(np.sum(real_mask))}")
    print(f"Actual fake examples:             {int(np.sum(fake_mask))}")
    print(f"Real incorrectly flagged fake:    {false_fake}")
    print(f"Fake incorrectly flagged real:    {false_real}")
    print(f"Correctly identified real:        {true_real}")
    print(f"Correctly identified fake:        {true_fake}")
    print(f"Inconclusive results:             {inconclusive}")
    print(f"Decisive coverage:                {np.mean(decisive) * 100:.2f}%")
    print(f"Accuracy on decisive results:     {decisive_accuracy * 100:.2f}%")
    print(f"Standard accuracy (50% cutoff):   {standard_accuracy * 100:.2f}%")
    print(f"Standard macro-F1 (50% cutoff):   {standard_f1:.4f}")


def main():
    print("Loading and cleaning the shared dataset...")

    df = load_and_preprocess_dataset(
        fake_path="data/Fake.csv",
        true_path="data/True.csv",
        min_words=20,
    )

    texts = df["text"].astype(str).to_numpy()
    labels = df["target"].astype(int).to_numpy()

    # Same fixed split for both models.
    _, X_test, _, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Shared test set: {len(X_test)} examples")
    print("Website thresholds: Real <= 40%, Fake >= 65%")

    for name, path in MODELS.items():
        evaluate(name, path, X_test, y_test)

    print("\nIMPORTANT:")
    print("This comparison uses the same examples for both models.")
    print("It is a genuinely held-out test only if these examples were")
    print("excluded from training both models. Dataset overlap or changes")
    print("in earlier preprocessing can affect the interpretation.")


if __name__ == "__main__":
    main()
