##########################################################################
# Procedural: Spiking Neuron Model (striatal -> GP -> thalamus -> cortex)
# Rule-based: Super-simple (if/else)
#
# Two responses (one for each model) -> one is chosen based on confidence
# Confidence: initial weight -> changes as a weighted reward history
#
# I am sorry for the extensive comments, otherwise I am worried I will not understand a thing 
##########################################################################

########## Packages ##########
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.animation import FuncAnimation
import pandas as pd

########## Global Variables/Arrrays ##########

# Timestamp for saving
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H_%M")  # format: 2025-05-22_15_30

# Simulation parameters
T = 3000
dt = 0.1
t = np.arange(0, T, dt)
n = t.shape[0]
n_trl = 100

# Arrays 
resp_neuron = np.zeros(n_trl) #Response - procedural pathway
resp_rulebased = np.zeros(n_trl) #Response - rulebased pathway
global_resp = np.zeros(n_trl) #Global response
confidence_neuron = np.zeros(n_trl) #Confidence - procedural pathway
confidence_rulebased = np.zeros(n_trl) #Confidence - rulebased pathway
cat = np.zeros(n_trl) # Category
rt = np.zeros(n_trl) # response time
resp_thresh = 1e10 # response threshold - how active should one pathway be compared to the other to create a response

# Visual input parameters
vis_dim = 100 #Dimensions 100x100
w_vis_msn = np.random.uniform(0.2, 0.4, (vis_dim**2, 2)) # starts narrow 
vis_act = np.zeros((vis_dim, vis_dim)) # visual activation - starts as nothing

# Initialize the confidence 
confidence_neuron[0] = 0.01 # initial confidence for procedural pathway
confidence_rulebased[0] = 0.99 # initial confidence for rulebased pathway

########## Procedural Pathway - Spinking neuron - Initialisation  ##########

# Variables and arrays
psp_amp = 3e5  # post-synaptic potential amplitude
psp_decay = 100  # post-synaptic potential decay time constant

# for each tiral not each time step
r = np.zeros(n_trl)
r_rulebased = np.zeros(n_trl) # reward for rulebased pathway
rpe = np.zeros(n_trl) # are sort of used instead of dopamine 
rpe_rulebased = np.zeros(n_trl) # same for rulebased pathway
alpha_critic = 0.38 # learning rate for critic
# make learning rate for neuron as array
# make learning rate for rulebased as an array

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

# For the plot 
coordinates = []

print("Starting loop...")

for trl in range(n_trl - 1):
    
    # Visual input categories
    # select stimulus (x, y) by sampling from a uniform distribution in [0, 100]
    x = np.random.uniform(0, 100)
    y = np.random.uniform(0, 100)
    coordinates.append((x, y))
    
    # assign category label
    if x < y:
        cat[trl] = 1
    else:
        cat[trl] = 2
        
    ########## Procedural Pathway ##########

    # compute visual response to stimulus vis_act as a 2D Gaussian
    x_grid, y_grid = np.meshgrid(np.arange(vis_dim), np.arange(vis_dim))
    vis_act = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * 6**2)) # The 10 controls for the width of the Gaussian - can be played around with
    vis_act *= 20 # the amplitude - can be played around with if the network is not responding enough or too much 

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
            resp_neuron[trl] = 1
            rt[trl] = i
            break
        elif (g[7, i] - g[3, i]) > resp_thresh:
            resp_neuron[trl] = 2
            rt[trl] = i
            break

    # pick a response if it hasn't happened already
    if rt[trl] == 0:
        resp_neuron[trl] = np.argmax(g[(3, 7), i]) + 1 # if neither pathwaw produced an action potentianl then pick the one that was most active
        rt[trl] = i
    
   
    ########## Rulebased Pathway - Super simple ##########
    
    # Choosing category 1 if the x coordinate is larger than 50 else category 2
    if x > 50:
        resp_rulebased[trl] = 1                    
    else:
        resp_rulebased[trl] = 2
   
    
    ########## Feedback to both systems ##########
    
    # Was the guess correct - Rulebased pathway
    if cat[trl] == resp_rulebased[trl]:
        r_rulebased[trl] = 1
    else:
        r_rulebased[trl] = 0
        
    # Confidence for rulebased
    rpe_rulebased[trl] = r_rulebased[trl] - confidence_rulebased[trl]

    #print(f"current confidence - rulebased {confidence_rulebased[trl]}")
    
    # !! I want to make the weighted mean instead over the last five instances maybe?
    # update the reward prediction
    weighted_mean = np.average(confidence_rulebased[:trl+1])
    #confidence_rulebased[trl + 1] = np.clip(weighted_mean + alpha_critic * rpe_rulebased[trl], 0, 1)
    confidence_rulebased[trl + 1] = np.clip(confidence_rulebased[trl] + alpha_critic * rpe_rulebased[trl], 0, 1)

    
    #### Procedural pathway
    # Was the guess correct - Procedural pathway
    if cat[trl] == resp_neuron[trl]:
        r[trl] = 1
    else:
        r[trl] = 0
    
    # reward prediction error
    rpe[trl] = r[trl] - confidence_neuron[trl]

    #print(f"current confidence - neuron {confidence_neuron[trl]}")
    # update the reward prediction
    confidence_neuron[trl + 1] = np.clip(confidence_neuron[trl] + alpha_critic * rpe[trl] , 0, 1)

    # Update visual-msn weights vs 3-factor RL rule 
    pre = vis_act.flatten() # flatten the 100x100 visual fiels to one long array
    post = g[(0, 4), :].sum(axis=1) 

    # implement / force hard laterial inhibition
    if resp_neuron[trl] == 1: # if the resp_neurononse was A 
        post[1] = 0 # force the B pathway to be inactive
    elif resp_neuron[trl] == 2:
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
    
    # for the video_plotting
    vis_act_over_time[trl] = vis_act  # visual input from that trial
    w_vis_msn_over_time_A[trl] = w_vis_msn[:, 0].reshape(vis_dim, vis_dim)
    w_vis_msn_over_time_B[trl] = w_vis_msn[:, 1].reshape(vis_dim, vis_dim)
    
    # Was the response correct - Rulebased pathway
    
    
    ########## Chosing a Global Response ##########
    
    if confidence_rulebased[trl] > confidence_neuron[trl]:
        global_resp[trl] = resp_rulebased[trl]
        print("Rulebased pathway chosen")
    else:   
        global_resp[trl] = resp_neuron[trl]
        print("Procedural pathway chosen")
    
# save some of the variables as a .csv to inpect 
df = pd.DataFrame({
    "trial": np.arange(n_trl),
    "Category": cat,
    "response_neuron": resp_neuron,
    "reward": r,
    "rpe": rpe, 
    "confidence_neuron": confidence_neuron,
    "response_rulebased": resp_rulebased,
    "reward_rulebased": r_rulebased,
    "rpe_rulebased": rpe_rulebased,
    "confidence_rulebased": confidence_rulebased
})  
 
# Save to CSV
df.to_csv(f"csv_files/csv_info_{timestamp}.csv", index=False)    
    
# ########## Plotting ##########

# # plot the confidence for both pathways over trials
# Assume confidence_neuron, confidence_rulebased, resp_neuron, resp_rulebased, cat, global_resp are defined
n_trials = len(cat)
correctness = np.array([1 if cat[i] == global_resp[i] else 0 for i in range(n_trials)])
corectness_neuron = np.array([1 if cat[i] == resp_neuron[i] else 0 for i in range(n_trials)])
corectness_rulebased = np.array([1 if cat[i] == resp_rulebased[i] else 0 for i in range(n_trials)])

df_correctness = pd.DataFrame({
    "corectness_neuron": corectness_neuron,
    "corectness_rulebased": corectness_rulebased,
    "correctness": correctness,
})
df_correctness.to_csv(f"csv_files/csv_correctness_{timestamp}.csv", index=False)   

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# --- 1. Confidence of both pathways ---
axs[0].plot(confidence_neuron, label="Procedural Confidence", color="blue")
axs[0].plot(confidence_rulebased, label="Rulebased Confidence", color="orange")
axs[0].plot(correctness, label="Correct Response Indicator", color="gray", linewidth=1)
axs[0].set_ylabel("Confidence")
axs[0].set_title("Confidence of Both Pathways + Correctness")
axs[0].set_ylim(-0.1, 1.2)
axs[0].legend()
axs[0].grid(True)

# --- 2. Procedural Pathway ---
#axs[1].plot(resp_neuron, label="Procedural Response", color="blue")
#axs[1].plot(np.where(resp_neuron == 2, 1, 0), label="Procedural Response", color="blue")
axs[1].plot(confidence_neuron, label="Confidence", linestyle="--", color="skyblue")
axs[1].plot(corectness_neuron, label="Correct Response ", color="blue", linewidth=1)
axs[1].set_ylabel("Response / Confidence")
axs[1].set_title("Procedural Pathway")
axs[1].set_ylim(-0.1, 1.2)
axs[1].legend()
axs[1].grid(True)

# --- 3. Rulebased Pathway ---
#axs[2].plot(resp_rulebased, label="Rulebased Response", color="orange")
#axs[2].plot(np.where(resp_rulebased == 2, 1, 0), label="Rulebased Response", color="orange")
axs[2].plot(confidence_rulebased, label="Confidence", linestyle="--", color="navajowhite")
axs[2].plot(corectness_rulebased, label="Correct Response ", color="orange", linewidth=1)
axs[2].set_ylabel("Response / Confidence")
axs[2].set_xlabel("Trial number")
axs[2].set_title("Rulebased Pathway")
axs[2].set_ylim(-0.1, 1.2)
axs[2].legend()
axs[2].grid(True)

# Final layout and save
plt.tight_layout()
plt.savefig(f"plots/{timestamp}_pathways_confidence_correctness.png")
plt.close()


# Scatterplot of coordinates

coordinates = np.array(coordinates) # Convert to NumPy array for easy plotting

plt.scatter(coordinates[:, 0], coordinates[:, 1])
plt.xlabel('x coordinates')
plt.ylabel('y coordinates')
plt.title('Visual stimnuli')
plt.savefig(f"plots/{timestamp}_coordinates_scatter_plot.png")
plt.close()


########## Video ##########
# Set up the plot
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# Initial images
im0 = ax[0].imshow(vis_act_over_time[0], vmin=0, vmax=1, animated=True)
im1 = ax[1].imshow(w_vis_msn_over_time_A[0], vmin=0, vmax=1, animated=True)
im2 = ax[2].imshow(w_vis_msn_over_time_B[0], vmin=0, vmax=1, animated=True)

# Add titles
ax[0].set_title("Visual Input")
ax[1].set_title("Prediction A (Weights to MSN A)")
ax[2].set_title("Prediction B (Weights to MSN B)")

def update(frame):
    im0.set_array(vis_act_over_time[frame])
    im1.set_array(w_vis_msn_over_time_A[frame])
    im2.set_array(w_vis_msn_over_time_B[frame])
    return im0, im1, im2

ani = FuncAnimation(fig, update, frames=n_trl, interval=100, blit=True)

# Save with timestamp
ani.save(f"videos/{timestamp}_vis_input_animation{n_trl}_{vis_dim}.mp4", writer="ffmpeg", fps=10)
