import torch.nn as nn
from hybrid_function import HybridFunction


class HybridLayer(nn.Module):
    def __init__(self, quantum_circuit, backend, *, shots, shift):
        super(HybridLayer, self).__init__()
        self.quantum_circuit = quantum_circuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)
