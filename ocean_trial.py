import numpy as np
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler

dwave_sampler_pegasus = DWaveSampler(solver={'topology__type': 'pegasus'})
props_pegasus = dwave_sampler_pegasus.properties

# Get total qubits - should be 24 * N * (N - 1)
total_qubits = props_pegasus['num_qubits']
print('total qbits = ', total_qubits)

# Get total number of inactive qubits
total_inactive = [i for i in range(total_qubits) if i not in dwave_sampler_pegasus.nodelist]
print('total inactive qbits = ', len(total_inactive))

# another way to compute the number of active qubits
active_qubits = dwave_sampler_pegasus.solver.num_active_qubits
print('active qbits, another way = ', active_qubits)

# This should convert the known inactive qubit indices to Pegasus coordinates.
inactive_pegasus_coord = [dnx.pegasus_coordinates(16).linear_to_pegasus(k) for k in total_inactive]
print('inactive qbits cords', inactive_pegasus_coord)