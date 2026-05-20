# ------------------------------------------------------------------------------
# Autore originale: Prof. Gabriele Russo Russo
# Riadattato da: Brunori Lorenzo per il progetto di tesi magistrale
# Data: Luglio 2025
# ------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np

def build_nn(output_dim, hidden_units, l2reg, input_dim = None):
    model = tf.keras.Sequential()
    if input_dim:
        model.add(tf.keras.Input(shape=(input_dim,)))  # Layer di input esplicito
    for hu in hidden_units:
        model.add(tf.keras.layers.Dense(hu, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(l2=l2reg)))
    model.add(tf.keras.layers.Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model


def expand_nn(old_nn, new_input_dim, new_output_dim, hidden_units, l2reg=0.0001):

    # Se non esiste modello, costruiscilo da zero
    if old_nn is None and hidden_units is not None:
        return build_nn(new_output_dim, hidden_units, l2reg, input_dim=new_input_dim)

    # Prendi pesi vecchi
    old_weights = [l.get_weights() for l in old_nn.layers]
    # Ricrea modello nuovo
    new_nn = build_nn(new_output_dim, hidden_units,l2reg, input_dim=new_input_dim)

    # Copia pesi input layer
    old_in_w, old_in_b = old_weights[1]
    new_in_w, new_in_b = new_nn.layers[1].get_weights()
    min_in = min(old_in_w.shape[0], new_in_w.shape[0])
    min_out = min(old_in_w.shape[1], new_in_w.shape[1])
    new_in_w[:min_in, :min_out] = old_in_w[:min_in, :min_out]
    new_in_b[:min_out] = old_in_b[:min_out]
    new_nn.layers[1].set_weights([new_in_w, new_in_b])

    # Copia pesi hidden invariati
    for i in range(2, len(new_nn.layers)-1):
        new_nn.layers[i].set_weights(old_weights[i])

    # Copia pesi output layer
    old_out_w, old_out_b = old_weights[-1]
    new_out_w, new_out_b = new_nn.layers[-1].get_weights()
    min_in = min(old_out_w.shape[0], new_out_w.shape[0])
    min_out = min(old_out_w.shape[1], new_out_w.shape[1])
    new_out_w[:min_in, :min_out] = old_out_w[:min_in, :min_out]
    new_out_b[:min_out] = old_out_b[:min_out]
    new_nn.layers[-1].set_weights([new_out_w, new_out_b])

    return new_nn


def expand_input_layer(old_layer, new_input_dim):
    old_w, old_b = old_layer.get_weights()
    old_input_dim, hidden_dim = old_w.shape

    # Nuova matrice pesi
    new_w = np.zeros((new_input_dim, hidden_dim), dtype=old_w.dtype)
    new_b = old_b.copy()

    # Copia pesi esistenti
    min_in = min(old_input_dim, new_input_dim)
    new_w[:min_in, :] = old_w[:min_in, :]

    # Nuovo layer con input aggiornato
    new_layer = tf.keras.layers.Dense(hidden_dim, activation='relu')
    new_layer.build((None, new_input_dim))
    new_layer.set_weights([new_w, new_b])

    return new_layer

def expand_output_layer(old_layer, new_output_dim):
    old_w, old_b = old_layer.get_weights()
    hidden_dim, old_output_dim = old_w.shape

    # Nuova matrice pesi
    new_w = np.zeros((hidden_dim, new_output_dim), dtype=old_w.dtype)
    new_w[:, :old_output_dim] = old_w

    new_b = np.zeros(new_output_dim, dtype=old_b.dtype)
    new_b[:old_output_dim] = old_b

    # Nuovo layer con input aggiornato
    new_layer = tf.keras.layers.Dense(new_output_dim)
    new_layer.build((None, hidden_dim))
    new_layer.set_weights([new_w, new_b])

    return new_layer