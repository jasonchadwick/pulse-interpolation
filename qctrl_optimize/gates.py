import numpy as np
import scipy.linalg as la

X = np.array([[0.0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1.0, 0], [0, 1]], complex)

def rx(theta):
    return la.expm(1j*theta/2*X)

def weyl(tx, ty, tz):
    return la.expm(-np.pi/2*1j*(tx*np.kron(X,X) + ty*np.kron(Y,Y) + tz*np.kron(Z,Z)))

def r3(tx, ty, tz):
    return la.expm(-np.pi/2*1j*(tx*X + ty*Y + tz*Z))