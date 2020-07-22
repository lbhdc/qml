import qiskit

from data_set import mnist
from hybrid_network import HybridNetwork
from quantum_runner import QuantumRunner
from test import test
from train import train


class QuantumCircuit(QuantumRunner):
    def __init__(self, n_qubits, backend, shots):
        self.backend = backend
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Quantum Circuit
        qubits = [i for i in range(n_qubits)]
        self._circuit.h(qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, qubits)
        self._circuit.measure_all()


if __name__ == "__main__":
    model = HybridNetwork(QuantumCircuit)
    model = train(model, mnist())
    test(model, mnist())
