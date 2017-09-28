numTilings = 4
    
def tilecode(in1, in2, tileIndices):
    # write your tilecoder here (5 lines or so)
    
    
def printTileCoderIndices(in1, in2):
    tileIndices = [-1] * numTilings
    tilecode(in1, in2, tileIndices)
    print('Tile indices for input (', in1, ',', in2,') are : ', tileIndices)

if __name__ == '__main__':
    printTileCoderIndices(-1.2, -0.07)
    printTileCoderIndices(-1.2, 0.07)    
    printTileCoderIndices(0.5, -0.07)    
    printTileCoderIndices(0.5, 0.07)
    printTileCoderIndices(-0.35, 0.0)
    printTileCoderIndices(0.0, 0.0)
    