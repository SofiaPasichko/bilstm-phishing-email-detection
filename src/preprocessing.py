import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(path="../data/phishing_email.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text_combined", "label"])

    duplicates_count = df.duplicated(subset="text_combined").sum()
    print(f"Дублікатів у датасеті: {duplicates_count}")

    df = df.dropna(subset=["text_combined", "label"])

    df = df.drop_duplicates(subset="text_combined").reset_index(drop=True)

    df["text_combined"] = df["text_combined"].str.lower()
    df["text_length"] = df["text_combined"].str.split().str.len()

    X = df["text_combined"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    pd.DataFrame({"text": X_train, "label": y_train}).to_csv("../data/train.csv", index=False)
    pd.DataFrame({"text": X_val, "label": y_val}).to_csv("../data/val.csv", index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv("../data/test.csv", index=False)

    print("\nРозподіл класів:")
    print("Train:")
    print(y_train.value_counts())

    print("\nValidation:")
    print(y_val.value_counts())

    print("\nTest:")
    print(y_test.value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    train_set = set(X_train)
    test_set = set(X_test)

    common = train_set.intersection(test_set)

    print("Спільних текстів:", len(common))
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")