import numpy as np
def ComputerEntropy(x, p_x):
      if(isinstance(p_x, float)):
            E = -p_x * np.log2(p_x)
            return x, E
#Test Case
x1, E1 = ComputerEntropy("532", 0.5)
x2, E2 = ComputerEntropy("531", 0.2)
x3, E3 = ComputerEntropy("530", 0.3)
print(x1, E1)
print(x2, E2)
print(x3, E3)
print("All H(x):", E1 + E2 + E3)