import jax
import jax.numpy as jnp
import utils
import h5py

file_name = 'clean'
num_qubits, num_layers = 4, 1
ini_key = 1234

cf_ideal = utils.Config(
    num_qubits = num_qubits,
    num_layers = num_layers,
    device_name = 'default.qubit',
    simulator_backend = None,
    shots = None,
    diff_method = 'backprop',
    interface = 'jax')
# allows shot noise and also device noise with the cell below
cf_noisy = utils.Config(
    num_qubits = num_qubits,
    num_layers = num_layers,
    device_name = 'qiskit.aer',
    simulator_backend = 'aer_simulator_statevector',
    shots = None,
    diff_method = 'parameter-shift',
    interface = 'jax')
# with open('IBMQ_token.txt', 'r') as f:
#     TOKEN = f.read().strip()
# provider = qiskit.IBMQ.enable_account(TOKEN)
# cf_noisy.load_noise_model('ibmq_manila', provider)

circuit_ideal, device_ideal = cf_ideal.make_device_circuit()
circuit_noisy, device_noisy = cf_noisy.make_device_circuit()

@jax.jit
def loss_ideal(params):
    return jnp.sum(circuit_ideal(params))
opt_ideal = utils.GradientDescentOptimizer(loss_ideal)
@jax.jit
def loss_noisy(params):
    return jnp.sum(circuit_noisy(params))
opt_noisy = utils.GradientDescentOptimizer(loss_noisy)

key = jax.random.PRNGKey(ini_key)
initializations = jax.random.uniform(key, (1000, num_layers, num_qubits, 3), minval=0., maxval=2*jnp.pi)

with h5py.File(f'{file_name}.hdf5', 'w') as f:
    results = [opt_ideal.optimize(ini, max_iter=200) for ini in initializations[:100]]
    f.create_dataset('results', data=results)
    for k, v in cf_noisy.items():
        f.attrs[k] = v
    with open(, 'r') as script:
        f.attrs['script'] = script.read()