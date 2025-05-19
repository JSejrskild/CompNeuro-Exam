##########################################################################
# Procedural: Spiking Neuron Model (striatal -> GP -> thalamus -> cortex)
# Rule-based: Super-simple (if/else)
#
# Two responses (one for each model) -> one is chosen based on confidence
# Confidence: initial weight -> changes as a weighted reward history
#
# Feedback:change pathways and confidence
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

# timing
T = 3000
dt = 0.1
t = np.arange(0, T, dt)
n = t.shape[0]
n_trl = 100

psp_amp = 3e5  # post-synaptic potential amplitude
psp_decay = 100  # post-synaptic potential decay time constant

# NOTE: Only inlucding direct pathway neurons for procedural category learning
# C, vr, vt, vp, a, b, c, d, k, E
izp = np.array([
    # A pathway neurons  - Response A
    [50, -80, -25, 40, 0.01, -20, -55, 150, 1, 0],      # str_d1 A
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7, 100],  # gpi A
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7, 100],  # thl A
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7, 0],    # ctx A
    # B pathway neurons  -  Response B
    [50, -80, -25, 40, 0.01, -20, -55, 150, 1, 0],      # str_d1 B
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7, 100],  # gpi B
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7, 100],  # thl B
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7, 0],    # ctx B
])

# NOTE: This is a new **very elegant** method of selecting neuron
# parameters to work with the activity propagation method used below
C, vr, vt, vpeak, a, b, c, d, k, E = izp.T

n_cells = izp.shape[0]

# network weights
w = np.zeros((n_cells, n_cells))

# A direct pathway
w[0, 1] = -1   # str_d1 -> gpi
w[1, 2] = -1   # gpi -> thl
w[2, 3] = 1    # thl -> ctx

# B direct pathway
w[4, 5] = -1   # str_d1 -> gpi
w[5, 6] = -1   # gpi -> thl
w[6, 7] = 1    # thl -> ctx

# laterial inhibition between str_d1 neurons
# Ensuring that only one str_d1 neuron is active at a time
w[0, 4] = -50.0  # str_d1 A -> str_d1 B
w[4, 0] = -50.0  # str_d1 B -> str_d1 A

# Weights: 
#  [[  0.  -1.   0.   0. -50.   0.   0.   0.]
#  [  0.   0.  -1.   0.   0.   0.   0.   0.]
#  [  0.   0.   0.   1.   0.   0.   0.   0.]
#  [  0.   0.   0.   0.   0.   0.   0.   0.]
#  [-50.   0.   0.   0.   0.  -1.   0.   0.]
#  [  0.   0.   0.   0.   0.   0.  -1.   0.]
#  [  0.   0.   0.   0.   0.   0.   0.   1.]
#  [  0.   0.   0.   0.   0.   0.   0.   0.]]

