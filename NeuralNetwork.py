import numpy as np
import tensorflow as tf
from tensorflow import keras



class NeuralNetwork():
    
    def __init__(self, weights = None) -> None:
        self.weights = weights
        self.createModel()

    def createModel(self):
        self.model = keras.models.Sequential([keras.Input((4,)),
                                              keras.layers.Dense(6, "relu", kernel_initializer="he_normal"),
                                              keras.layers.Dense(3, activation="softmax")])
        # Set the weights if they are given to the constructor.
        if self.weights is not None:
            # Get the indices of the dense layers.
            denseLayerIndices = [i for i, layer in enumerate(self.model.layers) if "dense" in layer.name]
            for i in range(len(self.weights)):
                self.model.layers[denseLayerIndices[i]].kernel = self.weights[i]

    def predict(self, x):
        prediction = self.model(tf.expand_dims(x, 0), training=False)
        direction = tf.argmax(tf.reshape(prediction, (3,))).numpy()
        if direction == 0:
            return "left"
        elif direction == 1:
            return None
        else:
            return "right"



if __name__ == "__main__":
    nn = NeuralNetwork()
    print(nn.model.layers[0].kernel.numpy())
    print(nn.model.layers[0].get_weights()[0])