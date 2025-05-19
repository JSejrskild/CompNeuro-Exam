## Reinforcement Learning Script 
## Aurthor: Matthew J. Crossley
## https://crossley.github.io/cogs3020/lectures/week_10/demo_rl.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

T = 1000  # Number of time steps
dt = 0.1  # Time step size
t = np.arange(0, T, dt) # Time vector
n = t.shape[0]  # Number of time steps
N = 200 # number of trials

psp_amp = 3e5 # post-synaptic potential amplitude
psp_decay = 100 # post-synaptic potential decay time constant

# C, vr, vt, vp, a, b, c, d, k, E
iz_params = np.array(
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7, 0]   # regular spiking
)

C = iz_params[0]
vr = iz_params[1]
vt = iz_params[2]
vp = iz_params[3]
a = iz_params[4]
b = iz_params[5]
c = iz_params[6]
d = iz_params[7]
k = iz_params[8]
E = iz_params[9]

# external input
I = np.zeros(n)
I[n//3:n//3*2] = 500

# define neurons
v_A = np.zeros((N, n))
u_A = np.zeros((N, n))
g_A = np.zeros((N, n))
spike_A = np.zeros((N, n))

v_B = np.zeros((N, n))
u_B = np.zeros((N, n))
g_B = np.zeros((N, n))
spike_B = np.zeros((N, n))

# initial conditions
v_A[:, 0] = vr # resting potential
u_A[:, 0] = b * vr # initial recovery variable
g_A[:, 0] = 0 # initial synaptic conductance

v_B[:, 0] = vr # resting potential
u_B[:, 0] = b * vr # initial recovery variable
g_B[:, 0] = 0 # initial synaptic conductance

# weight betwee neuron A and neuron B
w = np.zeros(n)
w[0] = 0.4

# critic parameters
response = np.zeros(N)
r_obtained = np.zeros(N)
r_predicted = np.zeros(N)
delta = np.zeros(N)

for trl in range(N-1): # iterate over trials

    for i in range(1, n): # iterate over time steps

        # update neuron A
        dvdt_A = ((k * (v_A[trl, i-1] - vr) * (v_A[trl, i-1] - vt)) - u_A[trl, i-1] + E + I[i-1]) / C
        dudt_A = a * (b * v_A[trl, i-1] - u_A[trl, i-1])
        dgdt_A = (-g_A[trl, i-1] + psp_amp * spike_A[trl, i-1]) / psp_decay

        # Update the state variables
        v_A[trl, i] = v_A[trl, i-1] + dvdt_A * dt
        u_A[trl, i] = u_A[trl, i-1] + dudt_A * dt
        g_A[trl, i] = g_A[trl, i-1] + dgdt_A * dt

        # Update spike records
        if v_A[trl,  i] >= vp:
            v_A[trl, i - 1] = vp
            v_A[trl, i] = vr
            u_A[trl, i] += d
            if i > n//10:
                spike_A[trl, i] = 1

        # update neuron B
        dvdt_B = ((k * (v_B[trl, i-1] - vr) * (v_B[trl, i-1] - vt)) - u_B[trl, i-1] + E + w[trl] * g_A[trl, i-1]) / C
        dudt_B = a * (b * v_B[trl, i-1] - u_B[trl, i-1])
        dgdt_B = (-g_B[trl, i-1] + psp_amp * spike_B[trl, i-1]) / psp_decay

        # Update the state variables
        v_B[trl, i] = v_B[trl, i-1] + dvdt_B * dt
        u_B[trl, i] = u_B[trl, i-1] + dudt_B * dt
        g_B[trl, i] = g_B[trl, i-1] + dgdt_B * dt

        # Update spike records
        if v_B[trl,  i] >= vp:
            v_B[trl, i - 1] = vp
            v_B[trl, i] = vr
            u_B[trl, i] += d
            if i > n//10:
                spike_B[trl, i] = 1

    # Determine response probability.
    resp_prob = 1.0 / (1.0 + np.exp(-w[trl] * 10 + 5))

    # determine network action
    if np.random.rand() < resp_prob:
        response[trl] = 1
    else:
        response[trl] = 0

    # determine obtained reward
    if response[trl] == 1:
        r_obtained[trl] = 1.0
    else:
        r_obtained[trl] = 0.0

    # If in the extinction phase no reward is given
    if trl > N//2:
        r_obtained[trl] = 0.0

    # determine reward prediction error
    delta[trl] = r_obtained[trl] - r_predicted[trl]

    # determine predicted reward
    alpha_pr = 0.05
    r_predicted[trl+1] = r_predicted[trl] + alpha_pr * (r_obtained[trl] - r_predicted[trl])

    # update weights with bio Hebbian learning rule modified for RL
    alpha = 1e-8; beta = 1e-8; lamb = 1e-5; theta = 1e2;

    pre = g_A[trl, :].sum()
    post = g_B[trl, :].sum()

    post_ltp_1 = np.heaviside(post - theta, 0)
    post_ltd_1 = np.heaviside(theta - post, 0)

    post_ltp_2 = 1.0 - np.exp(-lamb * (post - theta))
    post_ltd_2 = np.exp(-lamb * (theta - post))

    delta_w = 0.0
    if delta[trl] > 0:
        # LTP caused by strong post-synaptic activity and positive reward prediction error
        delta_w += alpha * pre * post_ltp_1 * post_ltp_2 * (1 - w[trl]) * delta[trl]
    else:
        # LTD caused by strong post-synaptic activity and negative reward prediction error
        delta_w -= alpha * pre * post_ltp_1 * post_ltp_2 * (1 - w[trl]) * np.abs(delta[trl])

    # LTD also caused by weak post-synaptic activity
    delta_w -= beta * pre * post_ltd_1 * post_ltd_2 * w[trl]

    w[trl+1] = w[trl] + delta_w

# plot the results as an animation
fig, ax = plt.subplots(4, 1, squeeze=False, figsize=(10, 10))

fig.subplots_adjust(hspace=0.5, wspace=0.5)

z0 = ax[0, 0].plot(np.arange(0, trl, 1), w[0:trl], color='C0')[0]
ax[0, 0].set_xlim(0, N)
ax[0, 0].set_ylim(0, 1)

ax1 = ax[1, 0]
ax2 = ax1.twinx()
z1 = ax1.plot(t, v_A[0, :], color='C0')[0]
z2 = ax2.plot(t, g_A[0, :], color='C1')[0]

ax3 = ax[2, 0]
ax4 = ax3.twinx()
z3 = ax3.plot(t, v_B[0, :], color='C0')[0]
z4 = ax4.plot(t, g_B[0, :], color='C1')[0]

z5 = ax[3, 0].plot(np.arange(0, N, 1), r_obtained, color='C0', label="Obtained Reward")[0]
z6 = ax[3, 0].plot(np.arange(0, N, 1), r_predicted, color='C1', label="Predicted Reward")[0]
z7 = ax[3, 0].plot(np.arange(0, N, 1), delta, color='C2', label="Reward Prediction Error")[0]

title = ax[0, 0].set_title('Trial 0')
ax[0, 0].set_ylabel('Synaptic Weight')
ax[0, 0].set_xlabel('Trial')
ax[1, 0].set_ylabel('Neuron A')
ax[2, 0].set_xlabel('Time (ms)')
ax[2, 0].set_ylabel('Neuron B')
ax[2, 0].set_xlabel('Time (ms)')
ax[3, 0].legend(loc='upper right')

def update(frame):
    title.set_text(f'Trial {frame}')
    z0.set_xdata(np.arange(0, frame, 1))
    z0.set_ydata(w[0:frame])
    z1.set_ydata(v_A[frame, :])
    z2.set_ydata(g_A[frame, :])
    z3.set_ydata(v_B[frame, :])
    z4.set_ydata(g_B[frame, :])
    z5.set_xdata(np.arange(0, frame, 1))
    z5.set_ydata(r_obtained[0:frame])
    z6.set_xdata(np.arange(0, frame, 1))
    z6.set_ydata(r_predicted[0:frame])
    z7.set_xdata(np.arange(0, frame, 1))
    z7.set_ydata(delta[0:frame])

    return title, z0, z1, z2, z3, z4

ani = FuncAnimation(fig, update, frames=N-1, blit=False, interval=100)
ani.save("../videos/bio_two_cell_hebb_rl.mp4", writer="ffmpeg", fps=10)