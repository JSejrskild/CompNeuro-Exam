##########################################################################
# Procedural: Spiking Neuron Model (striatal -> GP -> thalamus -> cortex)
# Rule-based: Super-simple (if/else)
#
# Two responses (one for each model) -> one is chosen based on confidence
# Confidence: initial weight -> changes as a weighted reward history
#
# Feedback:change pathways and confidence
##########################################################################

########## Packages ##########
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.animation import FuncAnimation



########## Global Variables/Arrrays ##########

# Simulation parameters
T = 3000
dt = 0.1
t = np.arange(0, T, dt)
n = t.shape[0]
n_trl = 100

# Arrays
resp = np.zeros(n_trl) #Response
cat = np.zeros(n_trl) # Category
rt = np.zeros(n_trl) # response time
resp_thresh = 1e10 # response threshold - how active should one pathway be compared to the other to create a response

# Visual input parameters
vis_dim = 100 #Dimensions 100x100
w_vis_msn = np.random.uniform(0.2, 0.4, (vis_dim**2, 2)) # starts narrow 
vis_act = np.zeros((vis_dim, vis_dim)) # visual activation - starts as nothing

# Visual input categories
 # select stimulus (x, y) by sampling from a uniform distribution in [0, 100]
x = np.random.uniform(0, 100)
y = np.random.uniform(0, 100)

########## Procedural Pathway - Spinking neuron ##########

# Variables and arrays
psp_amp = 3e5  # post-synaptic potential amplitude
psp_decay = 100  # post-synaptic potential decay time constant

# for each tiral not each time step
r = np.zeros(n_trl)
p = np.zeros(n_trl)
rpe = np.zeros(n_trl) # are sort of used instead of dopamine 
alpha_critic = 0.01 # learning rate for critic

# Synaptic plasticity learning rules
# how the striatal synapses learn
alpha_w = 5e-2
beta_w = 5e-2
gamma_w = 1e-2
theta = 0.1
lamb = 1e-5


# Initialising the neurons
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

# record keeping 
I_net = np.zeros((n_cells, n))  # net input to each neuron
v = np.zeros((n_cells, n)) # membrane potential
u = np.zeros((n_cells, n)) # recovery variable
g = np.zeros((n_cells, n)) # neurotransmitter release
spike = np.zeros((n_cells, n)) # spiking records 

# initial conditions
v[:, 0] = izp[:, 1]  # resting potential
u[:, 0] = izp[:, 5] * izp[:, 1]  # initial recovery variable
g[:, 0] = 0  # initial synaptic conductance

# for the video:
# Initialize arrays
vis_act_over_time = np.zeros((n_trl, vis_dim, vis_dim))
w_vis_msn_over_time_A = np.zeros((n_trl, vis_dim, vis_dim))
w_vis_msn_over_time_B = np.zeros((n_trl, vis_dim, vis_dim))

print("Starting simulation...")

for trl in range(n_trl - 1):
    
    # assign category label
    if x > y:
        cat[trl] = 1
    else:
        cat[trl] = 2

    # compute visual response to stimulus vis_act as a 2D Gaussian
    x_grid, y_grid = np.meshgrid(np.arange(vis_dim), np.arange(vis_dim))
    vis_act = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * 10**2)) # The 10 controls for the width of the Gaussian - can be played around with
    vis_act *= 10 # the amplitude - can be played around with if the network is not responding enough or too much 

    # reset I_net
    I_net.fill(0)
    v[:, 1:].fill(0)
    u[:, 1:].fill(0)
    g[:, 1:].fill(0)
    spike[:, 1:].fill(0)

    for i in range(1, n):  # iterate over time steps

        if i > n // 3:
            # Add visual input to msn neurons as the dot product of vis_act with w_vis_msn
            I_net[0, i - 1] = np.dot(vis_act.flatten(), w_vis_msn[:, 0]) # How much the visual input is affecting the striatal neurons (a answer)
            I_net[4, i - 1] = np.dot(vis_act.flatten(), w_vis_msn[:, 1]) # How much the visual input is affecting the striatal neurons (b answer)

        # Compute net input using matrix multiplication and remove self-connections - no neuron can excite itself 
        I_net[:, i - 1] += w.T @ g[:, i - 1] - np.diag(w) * g[:, i - 1]

        # Euler's method
        dvdt = (k * (v[:, i - 1] - vr) * (v[:, i - 1] - vt) - u[:, i - 1] + I_net[:, i - 1] + E) / C
        dudt = a * (b * (v[:, i - 1] - vr) - u[:, i - 1])
        dgdt = (-g[:, i - 1] + psp_amp * spike[:, i - 1]) / psp_decay

        v[:, i] = v[:, i - 1] + dvdt * dt
        u[:, i] = u[:, i - 1] + dudt * dt
        g[:, i] = g[:, i - 1] + dgdt * dt

        # spike detection - if the membrane potential exceeds the threshold set the membrane potential to the peak value
        mask = v[:, i] >= vpeak
        v[mask, i - 1] = vpeak[mask]
        v[mask, i] = c[mask]
        u[mask, i] += d[mask]
        spike[mask, i] = 1

        # response
        # g[3, i] = motor cortex A answer and g[7, i] = motor cortex B answer
        if (g[3, i] - g[7, i]) > resp_thresh:
            resp[trl] = 1
            rt[trl] = i
            break
        elif (g[7, i] - g[3, i]) > resp_thresh:
            resp[trl] = 2
            rt[trl] = i
            break

    # pick a response if it hasn't happened already
    if rt[trl] == 0:
        resp[trl] = np.argmax(g[(3, 7), i]) + 1 # if neither pathwaw produced an action potentianl then pick the one that was most active
        rt[trl] = i

    # feedback
    if cat[trl] == resp[trl]:
        r[trl] = 1
    else:
        r[trl] = 0

    # reward prediction error
    rpe[trl] = r[trl] - p[trl]

    # update the reward prediction
    p[trl + 1] = p[trl] + alpha_critic * rpe[trl]

    # Update visual-msn weights vs 3-factor RL rule 
    pre = vis_act.flatten() # flatten the 100x100 visual fiels to one long array
    post = g[(0, 4), :].sum(axis=1) 

    # implement / force hard laterial inhibition
    if resp[trl] == 1: # if the response was A 
        post[1] = 0 # force the B pathway to be inactive
    elif resp[trl] == 2:
        post[0] = 0

    post_ltp_1 = np.heaviside(post - theta, 0)
    post_ltd_1 = np.heaviside(theta - post, 0)

    # post_ltp_2 = 1.0 - np.exp(-lamb * (post - theta))
    # post_ltd_2 = np.exp(-lamb * (theta - post))

    # TODO: Simplifying hack to not deal with exponential decay which was 
    #       coming from overflow in the lines above
    post_ltp_2 = (1, 1)
    post_ltd_2 = (1, 1)

# The LTP and LTD can occur in different rates based on alpha and beta weights 
    if rpe[trl] > 0:
        # LTP caused by strong post-synaptic activity and positive reward prediction error
        dw_1A = alpha_w * pre * post_ltp_1[0] * post_ltp_2[0] * (1 - w_vis_msn[:, 0]) * rpe[trl]
        dw_1B = alpha_w * pre * post_ltp_1[1] * post_ltp_2[1] * (1 - w_vis_msn[:, 1]) * rpe[trl]
    else:
        # LTD caused by strong post-synaptic activity and negative reward prediction error
        dw_1A = beta_w * pre * post_ltp_1[0] * post_ltp_2[0] * (1 - w_vis_msn[:, 0]) * rpe[trl]
        dw_1B = beta_w * pre * post_ltp_1[1] * post_ltp_2[1] * (1 - w_vis_msn[:, 1]) * rpe[trl]

    # TODO: HACK TURN OFF FOR SIMPLICITY FOR NOW
    # LTD also caused by weak post-synaptic activity
    # dw_2 = gamma_w * pre * post_ltd_1 * post_ltd_2 * w_vis_msn
    dw_2A = 0
    dw_2B = 0

    # Apply the total weight change
    dwA = dw_1A + dw_2A
    dwB = dw_1B + dw_2B

    w_vis_msn[:, 0] += dwA
    w_vis_msn[:, 1] += dwB
    

########## Rulebased Pathway - Super simple ##########


########## Chosing a Global Response ##########


########## Plotting ##########





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

