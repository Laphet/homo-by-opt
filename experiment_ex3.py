import numpy as np

mat_A: np.array = np.load("data/b16r32cg-mat-A.npy")
mat_B: np.array = np.load("data/b16r32cg-mat-B.npy")
mat_c2f2: np.array = np.load("data/b16r32cg-mat-c2f2.npy")

print("A: size=(%d, %d), Max=%.5f, Min=%.5f, Norm=%.5f", *(mat_A.shape),
      np.max(mat_A), np.min(mat_A), np.linalg.norm(mat_A, ord=2))
print("B: size=(%d, %d), Max=%.5f, Min=%.5f, Norm=%.5f", *(mat_B.shape),
      np.max(mat_B), np.min(mat_B), np.linalg.norm(mat_B, ord=2))
print("c2f2: size=(%d, %d), Max=%.5f, Min=%.5f, Norm=%.5f", *(mat_c2f2),
      np.max(mat_c2f2), np.min(mat_c2f2), np.linalg.norm(mat_c2f2, ord=2))
