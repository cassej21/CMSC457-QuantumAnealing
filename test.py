# Quick example showing how to translate a QUBO matrix into
# D-Wave code.
# 
# The matrix in this problem is of the format
#
# | -5  2   4  0 |
# | 2  -3   1  0 |
# | 4   1  -8  5 |
# | 0   0   5 -6 |
#
# as described at https://arxiv.org/pdf/1811.11538.pdf

from dwave_qbsolv import *

q = {
        (0,0): -5,
        (1,1): -3,
        (2,2): -8,
        (3,3): -6,
        (0,1): 4,
        (0,2): 8,
        (1,2): 2,
        (2,3): 10
    }
response = QBSolv().sample_qubo(q)
print(response)