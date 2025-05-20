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

# We need a special model for visual cortex (because 
# category learning stimuli are fancy)
vis_dim = 10 #Should be 100 - but lower for less computational cost
w_vis_msn = np.random.uniform(0.2, 0.4, (vis_dim**2, 2)) # starts narrow 
vis_act = np.zeros((vis_dim, vis_dim)) # visual activation - starts as nothing

I_net = np.zeros((n_cells, n))  # net input to each neuron

# record keeping 
v = np.zeros((n_cells, n)) # membrane potential
u = np.zeros((n_cells, n)) # recovery variable
g = np.zeros((n_cells, n)) # neurotransmitter release
spike = np.zeros((n_cells, n)) # spiking records 

# initial conditions
v[:, 0] = izp[:, 1]  # resting potential
u[:, 0] = izp[:, 5] * izp[:, 1]  # initial recovery variable
g[:, 0] = 0  # initial synaptic conductance


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

resp = np.zeros(n_trl) #Response
cat = np.zeros(n_trl) # Category
rt = np.zeros(n_trl) # response time
resp_thresh = 1e10 # response threshold - how active should one pathway be compared to the other to create a response

for trl in range(n_trl - 1):

    # select stimulus (x, y) by sampling from a uniform distribution in [0, 100]
    x = np.random.uniform(0, 100)
    y = np.random.uniform(0, 100)

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
            I_net[0, i - 1] = np.dot(vis_act.flatten(), w_vis_msn[:, 0])
            I_net[4, i - 1] = np.dot(vis_act.flatten(), w_vis_msn[:, 1])

        # Compute net input using matrix multiplication and remove self-connections
        I_net[:, i - 1] += w.T @ g[:, i - 1] - np.diag(w) * g[:, i - 1]

        # Euler's method
        dvdt = (k * (v[:, i - 1] - vr) * (v[:, i - 1] - vt) - u[:, i - 1] + I_net[:, i - 1] + E) / C
        dudt = a * (b * (v[:, i - 1] - vr) - u[:, i - 1])
        dgdt = (-g[:, i - 1] + psp_amp * spike[:, i - 1]) / psp_decay

        v[:, i] = v[:, i - 1] + dvdt * dt
        u[:, i] = u[:, i - 1] + dudt * dt
        g[:, i] = g[:, i - 1] + dgdt * dt

        mask = v[:, i] >= vpeak
        v[mask, i - 1] = vpeak[mask]
        v[mask, i] = c[mask]
        u[mask, i] += d[mask]
        spike[mask, i] = 1

        # response
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
        resp[trl] = np.argmax(g[(3, 7), i]) + 1
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
    pre = vis_act.flatten()
    post = g[(0, 4), :].sum(axis=1)

    # implement / force hard laterial inhibition
    if resp[trl] == 1:
        post[1] = 0
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
    
# Next TODO: make the plots better 

# plot network activity per trial for diagnostic purposes
fig, ax = plt.subplots(2, 6, squeeze=False, figsize=(18, 4))

ax[0, 0].imshow(vis_act)
ax[1, 0].imshow(vis_act)

ax[0, 1].imshow(w_vis_msn[:, 0].reshape(vis_dim, vis_dim), vmin=0, vmax=1)
ax[1, 1].imshow(w_vis_msn[:, 1].reshape(vis_dim, vis_dim), vmin=0, vmax=1)

axx = ax[0, 2]
axx.plot(t, v[0, :], color='C0', label='str_d1 A')
axx2 = axx.twinx()
axx2.plot(t, g[0, :], color='C1')
ax[0, 2].legend()

axx = ax[1, 2]
axx.plot(t, v[4, :], label='str_d1 B')
axx2 = axx.twinx()
axx2.plot(t, g[4, :], color='C1')
ax[1, 2].legend()

axx = ax[0, 3]
axx.plot(t, v[1, :], label='gpi A')
axx2 = axx.twinx()
axx2.plot(t, g[1, :], color='C1')
ax[0, 3].legend()

axx = ax[1, 3]
axx.plot(t, v[5, :], label='gpi B')
axx2 = axx.twinx()
axx2.plot(t, g[5, :], color='C1')
ax[1, 3].legend()

axx = ax[0, 4]
axx.plot(t, v[2, :], label='thl A')
axx2 = axx.twinx()
axx2.plot(t, g[2, :], color='C1')
ax[0, 4].legend()

axx = ax[1, 4]
axx.plot(t, v[6, :], label='thl B')
axx2 = axx.twinx()
axx2.plot(t, g[6, :], color='C1')
ax[1, 4].legend()

axx = ax[0, 5]
axx.plot(t, v[3, :], label='ctx A')
axx2 = axx.twinx()
axx2.plot(t, g[3, :], color='C1')
ax[0, 5].legend()

axx = ax[1, 5]
axx.plot(t, v[7, :], label='ctx B')
axx2 = axx.twinx()
axx2.plot(t, g[7, :], color='C1')
ax[1, 5].legend()

plt.show()
plt.savefig('category_learning_activity.png')

# plot learning across trials
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
ax[0, 0].plot(np.arange(0, n_trl), r, color='C0', label='Obtained Reward')
ax[0, 0].plot(np.arange(0, n_trl), p, color='C1', label='Predicted Reward')
ax[0, 0].plot(np.arange(0, n_trl), rpe, color='C2', label='RPE')
plt.show()
plt.savefig('category_learning.png')