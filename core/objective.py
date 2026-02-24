import tensorflow as tf
from tensorflow.keras import backend as K

def cad_focal_loss(gamma=2.0, alpha=0.35):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)

        ce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = alpha * K.pow(1 - p_t, gamma) * ce

        return K.mean(fl)
    return loss
