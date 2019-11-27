import numpy as np 

EPSILON = 1.0e-5

def eigen_system(a: float, b: float, c: float, min_lambda: float):
    A = np.array([[a, c], [c, b]])
    s, d = a+b, a-b
    eigens = np.array([0.5*(s-np.sqrt(d**2+4.0*c**2)), 0.5*(s+np.sqrt(d**2+4.0*c**2))])
    if (np.min(eigens) < min_lambda):
        if (np.abs(c) < EPSILON):
            a = min_lambda if a < min_lambda else a
            b = min_lambda if b < min_lambda else b
        else:
            l1 = min_lambda if eigens[0] < min_lambda else eigens[0]
            l2 = min_lambda if eigens[1] < min_lambda else eigens[1]
            v1 = np.array([d-np.sqrt(d**2+4.0*c**2), 2*c])
            v2 = np.array([d+np.sqrt(d**2+4.0*c**2), 2*c])
            v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
            a = l1*v1[0]**2 + l2*v2[0]**2
            b = l1*v1[1]**2 + l2*v2[1]**2
            c = l1*v1[0]*v1[1] + l2*v2[0]*v2[1]
    A_mod = np.array([[a, c], [c, b]])
    return A, eigens, A_mod

A, eigens, A_mod = eigen_system(2., 3., 0., 2.5)
print(A)
print(eigens)
print(A_mod)

