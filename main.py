import numpy as np
from math import sqrt
import sys

A = np.array([[3,1,0],
              [1,3,1],
              [0,1,3]], dtype=float)


def q_multiply(cos, sin, A, i, j):
    tmp = np.sum([cos * A[i], - sin * A[j]], axis=0)
    A[j] = np.sum([sin * A[i], cos * A[j]], axis=0)
    A[i] = tmp


def qr_decomp(M):
    A = np.copy(M)
    Q = np.identity(A.shape[0], dtype=float)
    print(f'\nQR ALGORITHM:')
    print(f'cond A: {np.linalg.cond(A)}')
    print(f'#\tpos\t\t\t\tcos\t\t\t\t\tsin')
    counter = 0
    for i in range(A.shape[1]):
        for j in range(i + 1, A.shape[0]):
            c, s = map(float, [0, 0])

            counter += 1
            if max(A[i][i], A[j][i]) == 0:
                print(f'{counter}\t({j},{i})\tmissed')
                continue

            lng = sqrt(A[i][i]**2 + A[j][i]**2)
            c = A[i][i] / lng
            s = - A[j][i] / lng
            q_multiply(c, s, A, i, j)
            A[j][i] = 0
            q_multiply(c, s, Q, i, j)
            print(f'{counter}\t({j},{i})\t{c}\t{s}')
    Q = Q.T
    print(f'\nMatrix Q:\n{Q}\nMatrix R:\n{A}')
    print(f'Matrix Q and R multiplication:\n{np.dot(Q, A)}')
    print(f'\ncond R: {np.linalg.cond(A)}')
    print(f'Quantity of iterations: {counter}')
    print(f'END OF QR\n\n')
    return Q, A


def qr_det(R):
    return (np.diag(R)).prod()


def det(A):
    print(F'DETERMINANT ALGORITHM:')
    print(f'Matrix:\n{A}')
    tmp = qr_det(qr_decomp(A)[1])
    print(f'Mine determinant: {tmp}')
    print(f'NumPy determinant: {np.linalg.det(A)}')
    print(f'END OF DETERMINANT ALGORITHM\n')
    return tmp


def inverse(X):
    print(f'INV:')
    std = sys.stdout
    sys.stdout = None
    Q, R = qr_decomp(X)
    size = X.shape[0]
    deter = qr_det(R)
    if deter == 0:
        print(f'\nTHE MATRIX IS SINGULAR, THE INVERSE DOES NOT EXIST\n')
        return None
    rev = np.array([[(-1)**(k + m) * det(np.array([[R[j][i]
                for j in range(size) if j != k]
                    for i in range(size) if i != m]))
                        for k in range(size)]
                            for m in range(size)])
    rev = rev / deter
    mrx = np.dot(rev, Q.T)
    sys.stdout = std
    print(f'\nMine inverse matrix:\n{mrx}')
    print(f'NumPy inverse matrix:\n{np.linalg.inv(X)}\n')
    return mrx


def qr_method(A):
    orig_stdout = sys.stdout
    f = open('qr_det.txt', 'w')
    sys.stdout = f

    print(f'QR DETERMINANT|MAMEDOV VALENTIN[17144]\n')
    print(f'Matrix A:\n{A}\n')
    det(A)
    inverse(A)

    sys.stdout = orig_stdout
    f.close()


qr_method(A)
