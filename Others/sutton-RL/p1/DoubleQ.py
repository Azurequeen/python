import blackjack
from pylab import *

Q1 = ... # NumPy array of correct size
Q2 = ... # NumPy array of correct size

def learn(alpha, eps, numTrainingEpisodes):
    returnSum = 0.0
    for episodeNum in range(numTrainingEpisodes):
        G = 0
        ...
        # Fill in Q1 and Q2
        print("Episode: ", episodeNum, "Return: ", G)
        returnSum = returnSum + G
        if episodeNum % 10000 == 0 and episodeNum != 0:
            print("Average return so far: ", returnSum / episodeNum)

def evaluate(numEvaluationEpisodes):
    returnSum = 0.0
    for episodeNum in range(numEvaluationEpisodes):
        G = 0
        ...
        # Use deterministic policy from Q1 and Q2 to run a number of
        # episodes without updates. Return average return of episodes.
        returnSum = returnSum + G
    return returnSum / numEvaluationEpisodes

