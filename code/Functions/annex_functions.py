from Library.import_library import *


def print_matrix(matrix, texte):
    print(texte)
    try:
        n, m = matrix.shape
    except:
        n = len(matrix)

    for i in range(n):
        print(matrix[i, :])
