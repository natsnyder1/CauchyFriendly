import numpy as np

A = np.array([-0.8183301800, 0.4994945217, 0.2583404066, 
-0.9146071802, -1.0964502735, 2.0441714662, 
0.2146585676, 0.4906267934, 1.3553690103]).reshape((3,3))


B = np.array([-0.4650540760, 0.8996140972, 0.1783884052, 
0.3838444179, 2.0142001403, -1.4473637004, 
0.8660659975, 1.8096649606, 2.2045673865]).reshape((3,3))

c = np.array([
-0.8806707611, 
-0.1830279329, 
-0.1986892353])

print("A:\n")
print(A)
print("B:\n")
print(B)
print("c:\n")
print(c)
C = A @ B
C = B @ C
print("B @ A @ B\n")
print(C)
print("A @ B:\n")
print(A @ B)
print("A @ c:\n")
print(A @ c)
print("A * c:\n")
print(A * c.reshape((3,1)))
print("A * c.T:\n")
print(A * c.T)
print("A.I @ B:\n")
print(np.linalg.inv(A) @ B)
print("A+B:\n")
print(A+B)
print("A-B:\n")
print(A - B)
print("A @ B - B.T @ A:\n")
print((A @ B) - (B.T @ A))
print("A @= A:\n")
A = A @ A
print(A)
print("B += B:\n")
B += B
print(B)
print("c & c.T\n")
print(np.outer(c,c))

D = A * 2.0 + c.reshape((3,1)) * B 
E = D * 2.0
print("D:\n")
print(D)
print("E:\n")
print(E)
E = A @ B
print(E)
E = E @ D
print(E)

# Solve Triangular

Z = np.array([1,0,0,1,2,0,1,2,3]).reshape((3,3))
W = np.arange(12).reshape(3,4)
print(Z)
print(W)
print(np.linalg.inv(Z) @ W)
print(np.linalg.inv(Z.T) @ W)

foo = np.linalg.inv((np.linalg.inv(Z) @ C).T @ A + B)
foo = foo @ A
foo = foo @ (B + D)
print(foo)