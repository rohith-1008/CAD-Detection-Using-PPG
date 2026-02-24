import tensorflow as tf
from tensorflow.keras import layers, models, Input

class AttentionPooling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.att = layers.Dense(1)

    def call(self, x):
        w = tf.nn.softmax(self.att(x), axis=1)
        return tf.reduce_sum(x * w, axis=1)


def build_cnn_bilstm(ppg_shape, clinical_shape):
    ppg_input = Input(shape=ppg_shape, name="ppg_input")

    x = layers.Conv1D(32, 5, padding="same", activation="relu")(ppg_input)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2
        )
    )(x)

    x = layers.Dropout(0.4)(x)
    x = AttentionPooling()(x)

    clin_input = Input(shape=clinical_shape, name="clinical_input")
    y = layers.Dense(16, activation="relu")(clin_input)
    y = layers.Dropout(0.3)(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.4)(z)

    output = layers.Dense(1, activation="sigmoid")(z)

    return models.Model(
        inputs=[ppg_input, clin_input],
        outputs=output,
        name="CNN_BiLSTM_Attention_CAD"
    )
