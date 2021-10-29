import numpy as np
import tensorflow as tf
import random
from NeuralNetwork import NeuralNetwork
from pong_game import playPong



class GeneticAlgorithm():

    def __init__(self, populationSize = 100, nGenerations = 100) -> None:
        self.populationSize = populationSize

        self.createPopulation()
        for i in range(nGenerations):
            print(f"Current Generation: {i}")
            self.apply()
            if i < (nGenerations-1):
                self.crossoverAndMutate()
            print(f"Best score: {max(self.scores)}")
            print(f"Total score: {sum(self.scores)} \n")
        
        # Sort the models based on their scores.
        self.population = [nn for _, nn in sorted(zip(self.scores, self.population), key=lambda pair: pair[0], reverse=True)]
        self.population[0].model.save("./models/model.h5")

    def createPopulation(self) -> None:
        self.population = []
        for _ in range(self.populationSize):
            self.population.append(NeuralNetwork())

    def apply(self):
        self.scores = []
        for i in range(self.populationSize):
            # print(f"Current model {i} of {len(self.population)}")
            self.scores.append(playPong(controlFunction=self.population[i].predict, visualize=False))

    def crossoverAndMutate(self):
        # Sort the population based on their scores.
        self.population = [nn for _, nn in sorted(zip(self.scores, self.population), key=lambda pair: pair[0], reverse=True)]
        # Create a list with the number of occurences based on the scores of the models.
        choiceList = []
        for i, score in enumerate(self.scores):
            choiceList += score*[i]
        newPopulation = []
        newPopulation.extend(self.population[:round(0.1*self.populationSize)])
        for _ in range(round(0.1*self.populationSize), self.populationSize):
            # Randomly select two models.
            nn1 = self.population[random.choice(choiceList)]
            nn2 = self.population[random.choice(choiceList)]
            # Find the indices of the dense layers.
            denseLayerIndices = [i for i, layer in enumerate(nn1.model.layers) if "dense" in layer.name]
            # Loop over every dense layer in the networks.
            newWeights = []
            for layerIndex in denseLayerIndices:
                kernel1 = nn1.model.layers[layerIndex].kernel.numpy()
                kernel2 = nn2.model.layers[layerIndex].kernel.numpy()
                shape = kernel1.shape
                # Loop over every neuron in the layer.
                for j in range(shape[1]):
                    # Create a random split entry.
                    splitIndex = random.randint(0, shape[0]-1)
                    if splitIndex > 0:
                        kernel1[:splitIndex, j] = kernel2[:splitIndex, j]
                # Mutate the weights.
                newWeights.append(self.mutate(kernel1))
            newPopulation.append(NeuralNetwork(newWeights))
        self.population = newPopulation



    def mutate(self, kernel):
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if random.uniform(0, 1) < 0.2:
                    kernel[i, j] = random.randrange(-1, 1)
        return tf.Variable(kernel)



if __name__ == "__main__":
    ga = GeneticAlgorithm(100, 30)