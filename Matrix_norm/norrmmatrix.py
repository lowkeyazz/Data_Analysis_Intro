import numpy as np
from numpy import linalg as LA
from math import sqrt

def normatrice():
    a=np.random.uniform(-1,1,28)
    A=np.empty((7,4))
    AT=np.empty((4,7))
    P=np.empty((4,4))
    PT=np.empty((7,7))
    v=0
    for i in range(7):
        for j in range(4):
            A[i][j]=a[v]
            AT[j][i]=a[v]
            v+=1
    P=AT.dot(A)
    PT=A.dot(AT)
    Sp , w =LA.eig(P)
    SpT , w =LA.eig(PT)
    f=max(Sp)
    ft=max(SpT)
    print("Les valeurs propres de la matrice sont :",Sp)
    print("la norme de la matrice A est : ",sqrt(f))
    print("la norme de la matrice A transpose est : ",sqrt(ft))
normatrice()
normatrice()
normatrice()
"""On peut dire que ||A|| et ||AT|| sont égaux"""