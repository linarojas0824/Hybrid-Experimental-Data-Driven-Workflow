"""
This script is adapted from:

Author: Taeyeop Kim and Dongwoo Lee
Title: Unveiling Key Descriptors for Electrical Resistivity of Alloys 
Using High-Throughput Experiments and Explainable AI
Repository: https://github.com/TaeyeopK/ML-for-Electrical-Resistivity-of-Alloys.git
Year: 2025
"""

import tensorflow as tf

class Net():
    def __init__(self, input_dim, num_dense_layers, num_dense_nodes):
        self.input_dim = input_dim
        self.num_dense_layers = num_dense_layers
        self.num_dense_nodes = num_dense_nodes
        self.model = self.build()

    def build(self):
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = inputs
        
        for i in range(self.num_dense_layers):
            name = f'layer{i}'
            x = tf.keras.layers.Dense(self.num_dense_nodes, activation='relu',
                          name=name)(x)
            x = tf.keras.layers.Dropout(0.3)(x)

        output = tf.keras.layers.Dense(1, activation='linear', name='r')(x)
        
        ann = tf.keras.Model(inputs, output, name='ANN')

        return ann

    @tf.function
    def train_step(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            output = self.model(x)
            loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            loss = tf.reduce_mean(loss_fn(output, y))**0.5

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def validation_step(self, x, y):
        output = self.model(x)
        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss = tf.reduce_mean(loss_fn(output, y))**0.5
        return loss

    def train(self, num_epochs, batch_train, batch_val, optimizer):
        best_error = 100
        history = []

        for epoch in range(num_epochs):
            train_losses = []

            for x, y in batch_train:
                loss = self.train_step(x, y, optimizer)
                train_losses.append(loss)

            train_loss = tf.reduce_mean(train_losses)

            val_losses = []
            for x, y in batch_val:
                loss = self.validation_step(x, y)
                val_losses.append(loss)

            val_loss = tf.reduce_mean(val_losses)

            history.append(
                {"train_loss": train_loss.numpy(), "val_loss": val_loss.numpy()})

            if val_loss < best_error:
                best_error = val_loss
                self.model.save(f'./model/ANN_customized.h5')

            print('epoch: {}/{}, train loss: {:.4f}, val loss: {:.4f}'.format(
                epoch + 1, num_epochs, train_loss.numpy(), val_loss.numpy()))

        return history