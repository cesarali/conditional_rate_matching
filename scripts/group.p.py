# %% [markdown]
# #### Question 1a:
# Plot the x-component of your solution over the time interval [0,10] similarly to Figure 1.3 in the textbook.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def lorenz_f(z):
    x, y, z = z
    return np.array([10 * (y - x), x * (28 - z) - y, x * y - (8/3) * z])

def g_n(g, a):
    new_g = np.zeros((1,3))
    for i in range(3):
        if g[i] < 0 and g[i] >= -a/2: new_g[i] = 1.99999 * g[i] + a / 2
        else: -1.99999 * g[i] + a / 2
    return new_g

def lorenz63_with_g(z0, g0, dt, T, a):
    N = int(T / dt)
    t = np.linspace(0, T, N)
    z = np.zeros((N, 3))
    g = np.zeros((N, 3))
    z[0] = z0
    g[0] = g0
    for i in range(1, N):
        g[i] = g_n(g[i-1], a)
        z[i] = z[i-1] + dt * (lorenz_f(z[i-1]) + g[i-1])
    return t, z

# Constants and initial conditions
dt = 0.01
T = 10
a = 1 / np.sqrt(dt)
z0 = np.array([-0.587, -0.563, 16.870])
g0 = np.array([a * (np.sqrt(2) - 0.5), a * (np.sqrt(3) - 0.5), a * (np.sqrt(5) - 0.5)]).reshape((1,3))

# Solve the system
t, z = lorenz63_with_g(z0, g0, dt, T, a)

# Plotting the x-component
plt.plot(t, z[:, 0])
plt.title('Lorenz-63 System with Modifications')
plt.xlabel('Time')
plt.ylabel('x-component')
plt.grid(True)
plt.show()


# %% [markdown]
# #### Question 1b:
# 
# Store the resulting reference trajectory in time intervals of ∆tout = 0.05 over 4000 cycles (i.e., between t = 0 and t = 200) in a file for later use in other examples. Print the mean and standard deviation of the three matrix rows corresponding to x, y, and z. (Do not store the system state from every single timestep as this becomes very inefficient, even for low dimensional problems; it is much better to overwrite the state vector on each timestep, and take a copy of the vector when you need to store it. The resulting data set should be stored in a matrix of size 3 × 4001.)

# %%
def updated_g_n(g, a):
    # g is 3x1
    new_g = np.zeros((3,1))
    
    for i in range(3):
        if g[i] < 0 and g[i] >= -a/2: new_g[i] = 1.99999 * g[i] + a / 2
        else: new_g[i] = -1.99999 * g[i] + a / 2
    
    return new_g

def updated_lorenz_f(z):
    x, y, z = z # z is 3x1
    return np.array([10 * (y - x), x * (28 - z) - y, x * y - (8/3) * z])


def updated_lorenz63_with_g(z0, g0, dt, T, a):
    N = int(T / dt)
    t = np.linspace(0, T, N)
    z = z0.T.reshape((3,1))
    g = g0.T
    
    final = np.zeros((3,4001))
    index = 0
    for i in range(1, N):
        
        z = z + dt * (updated_lorenz_f(z) + g)
        
        g = updated_g_n(g, a)

        if i % 5 == 0: 
            final[:,index] = z[:,0]
            index += 1

    return t, final

T = 200

t, final = updated_lorenz63_with_g(z0, g0, dt, T, a)

np.savetxt("trajectory.txt", final)


# %%
print("Mean of x,y,z:" , np.mean(final, axis=1))
print("STD of x,y,z:" , np.std(final, axis=1))

"""
# %% [markdown]
# #### Question 2a:
# 
# Plot the observed x-components and the corresponding measurement errors over the time interval [0,10] similarly to Figure 1.3 in the textbook.

# %%
x_org = np.loadtxt("trajectory.txt")[0,:]


# %%
a = 4
n_steps = 4001*20
E = np.zeros((n_steps,1))
E[0] = a * (np.sqrt(2) - 0.5)

for i in range(1,n_steps):
    if E[i-1] > -a/2 and E[i-1] < 0: E[i] = 1.99999*E[i-1] + a/2
    else: -1.99999*E[i-1] + a/2



# %%
E_summed = E.reshape(-1, 20).sum(axis=1) / 20


# %%
x_obs = x_org + 1/20 * E_summed


# %%
time = np.linspace(0, 10, int(10 / 0.05) + 1)

plt.figure(figsize=(12, 6))
plt.plot(time, x_org[:201], label='True x-component', linestyle='--')
plt.scatter(time, x_obs[:201], color='r', label='Observed x-component', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('x-component')
plt.title('True vs Observed x-components')
plt.legend()
plt.grid(True)
plt.show()

# Plot the measurement errors over the time interval [0, 10]
errors = x_obs[:201] - x_org[:201]
plt.figure(figsize=(12, 6))
plt.plot(time, errors, label='Measurement Error', color='orange')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Measurement Errors in x-component')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# #### Question 3

# %%

# Function to perform linear extrapolation
def linear_extrapolation(x, y, future_time):
    # Fit linear regression model
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    # Extrapolate future values
    future_x = np.array([x[-1] + future_time])
    future_y = model.predict(future_x.reshape(-1, 1))

    return future_y[0]

def compute_rmse(observed, forecast):
    return np.sqrt(mean_squared_error(observed, forecast))

# Function to calculate time-averaged RMSE
def time_averaged_rmse(observed, forecast, forecast_interval):
    rmse_values = []
    for i in range(len(observed) - 1):
        forecast_value = linear_extrapolation(np.arange(i + 1), observed[:i + 1], forecast_interval)
        rmse = compute_rmse(observed[i + 1:], np.repeat(forecast_value, len(observed[i + 1:])))
        rmse_values.append(rmse)
    return np.mean(rmse_values)

# Example observations from Exercise 2
# Replace this with your actual data
observed_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Forecast intervals
delta_t_out_1 = 0.05
delta_t_out_2 = 3 * delta_t_out_1

# Perform extrapolation and compute RMSE
forecast_1 = linear_extrapolation(np.arange(len(observed_data)), observed_data, delta_t_out_1)
forecast_2 = linear_extrapolation(np.arange(len(observed_data)), observed_data, delta_t_out_2)

# Compute time-averaged RMSE
time_avg_rmse_1 = time_averaged_rmse(observed_data, forecast_1, delta_t_out_1)
time_avg_rmse_2 = time_averaged_rmse(observed_data, forecast_2, delta_t_out_2)

print("Forecast for delta_t_out =", delta_t_out_1, ": ", forecast_1)
print("Time-averaged RMSE for delta_t_out =", delta_t_out_1, ": ", time_avg_rmse_1)

print("Forecast for delta_t_out =", delta_t_out_2, ": ", forecast_2)
print("Time-averaged RMSE for delta_t_out =", delta_t_out_2, ": ", time_avg_rmse_2)

"""
