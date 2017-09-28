import blackjack
from pylab import *

def run(numEvaluationEpisodes):
    returnSum = 0.0
    for episodeNum in range(numEvaluationEpisodes):
        G = 0
        ...
        print("Episode: ", episodeNum, "Return: ", G)
        returnSum = returnSum + G
    return returnSum / numEvaluationEpisodes

