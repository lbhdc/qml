import numpy as np
import qiskit


class QuantumRunner:


    def run(self, thetas):
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas]
        )

        result = job.result().get_counts(self._circuit)

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)

        return np.array([expectation])