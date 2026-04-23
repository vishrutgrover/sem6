# WAP to find relationship between pixels (4 point, 8 point, diagonal)
import numpy as np

n = int(input("Enter n for nxn matrix: "))
matrix = np.random.randint(0, 256, (n, n), dtype=np.uint8)
print("Matrix:")
print(matrix)

row = int(input(f"Enter row (0-{n-1}): "))
col = int(input(f"Enter col (0-{n-1}): "))

neigh4 = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
neigh8 = [(row-1, col-1), (row-1, col), (row-1, col+1), (row, col-1), (row, col+1), (row+1, col-1), (row+1, col), (row+1, col+1)]
neighdiag = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]

neigh4 = [(r, c) for r, c in neigh4 if 0 <= r < n and 0 <= c < n]
neigh8 = [(r, c) for r, c in neigh8 if 0 <= r < n and 0 <= c < n]
neighdiag = [(r, c) for r, c in neighdiag if 0 <= r < n and 0 <= c < n]

print(f"\n4-point neighbors of ({row}, {col}): {neigh4}")
print(f"4-point pixel values: {[matrix[r, c] for r, c in neigh4]}")

print(f"\n8-point neighbors of ({row}, {col}): {neigh8}")
print(f"8-point pixel values: {[matrix[r, c] for r, c in neigh8]}")

print(f"\nDiagonal neighbors of ({row}, {col}): {neighdiag}")
print(f"Diagonal pixel values: {[matrix[r, c] for r, c in neighdiag]}")

print(f"\nPixel value at ({row}, {col}): {matrix[row, col]}")