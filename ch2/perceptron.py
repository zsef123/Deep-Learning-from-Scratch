import numpy as np

# Single-Layer-Perceptrons
NAND = (np.array([-0.5, -0.5]),  0.7)
OR   = (np.array([0.5,   0.5]), -0.2)
AND  = (np.array([0.5,   0.5]), -0.7)
def gate(x , x1, gateType):
    (ws, b) = gateType
    xs = np.array([x, x1])
    tmp = np.sum(xs*ws) + b
    return 0 if tmp<=0 else 1

# Multi-Layer-Perceptron
def xorGate(x, x1):
    a = gate(x, x1, NAND)
    b = gate(x, x1, OR)
    return gate(a, b, AND)

if __name__ == "__main__":
    print(gate(0, 1, NAND))
    print(xorGate(1,1))