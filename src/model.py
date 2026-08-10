from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)


def create_model(max_words, embedding_dim=128):
    model = Sequential([
        Embedding(
            input_dim=max_words,
            output_dim=embedding_dim
        ),

        Bidirectional(
            LSTM(128, return_sequences=True)
        ),

        Dropout(0.3),

        Bidirectional(
            LSTM(64)
        ),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model