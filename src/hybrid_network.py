import numpy as np
import qiskit
import torch
import torch.nn as nn
import torch.nn.functional as F
from hybrid_layer import HybridLayer


class HybridNetwork(nn.Module):
    def __init__(self, quantum_circuit, quantum_backend="qasm_simulator"):
        super(HybridNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = HybridLayer(
            quantum_circuit,
            qiskit.Aer.get_backend(quantum_backend),
            shots=100,
            shift=np.pi / 2
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)
