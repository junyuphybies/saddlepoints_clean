import pennylane as qml
import jax
import jax.numpy as jnp
from tqdm import tqdm
from qiskit.providers.aer.noise import NoiseModel
from dataclasses import dataclass


@dataclass
class Config:
    device_name: str
    num_qubits: int
    num_layers: int
    simulator_backend: str
    shots: int
    diff_method: str
    interface: str
    
    def __post_init__(self):
        pass

    def load_noise_model(self, ibm_backend_name, provider):
        backend = provider.get_backend(ibm_backend_name)
        self.noise_model = NoiseModel.from_backend(backend)
        self.noise_model_name = f'Noise model from {ibm_backend_name}'
    
    def make_device_circuit(self):
        if hasattr(self, 'noise_model'):
            noise_model = self.noise_model
        else:
            noise_model = None
        if self.simulator_backend is not None:
            device = qml.device(
                self.device_name,
                wires=self.num_qubits,
                backend=self.simulator_backend,
                shots=self.shots,
                noise_model=noise_model
            )
        else:
            device = qml.device(
                self.device_name,
                wires=self.num_qubits,
                shots=self.shots,
            )
        @qml.qnode(device, diff_method=self.diff_method, interface=self.interface)
        def circuit(parameters):
            qml.StronglyEntanglingLayers(parameters, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        return circuit, device
    

class GradientDescentOptimizer:
    def __init__(self, loss_function):
        self.grad = jax.value_and_grad(loss_function)

    def optimize(self, initialization, artificial_noise=None, step_size=0.2, key_seed=0, max_iter=100):
        key = jax.random.PRNGKey(key_seed)
        loss_history, grad_history, param_history = [], [], [initialization]
        for _ in tqdm(range(max_iter)):
            p = param_history[-1]
            l, g = self.grad(p)
            if artificial_noise is None:
                param_history.append(p - step_size*g)
            else:
                k, key = jax.random.split(key)
                perturbation = jax.random.uniform(key, p.shape, minval=-1, maxval=1)
                param_history.append(p - step_size*g + artificial_noise * perturbation)
            loss_history.append(l)
            grad_history.append(g)
            if jnp.linalg.norm(g) / g.size < 1e-4:
                break
        return loss_history, grad_history, param_history
