#!/usr/bin/env python
# coding: utf-8
# Exercise 1

import numpy as np
import matplotlib.pyplot as plt

# Parameters
E0 = -65  # Resting potential (mV)
tau = 30  # Time constant (ms)
r = 10    # Membrane resistance (MOhm)
g = 1 / r # Membrane conductance
C = tau / r
Vt = -55  # Threshold potential (mV)

# Simulation settings
dt = 0.01
tend = 300  # ms
t = np.arange(0, tend + dt, dt)
L = len(t)

# Initialize membrane potential and input current
V = np.zeros((L,))
V[0] = -65  # Initial value (mV)
Imax = 4.0  # Maximum current amplitude (nA)
I = np.abs(Imax * np.sin(np.pi * t / tend))

# Simulate neuron behavior
idx_spike = []
for k in np.arange(L - 1):
    Vinf = E0 + r * I[k]
    V[k + 1] = V[k] + (Vinf - V[k]) * dt / tau  # Euler's method
    if V[k + 1] > Vt:
        V[k + 1] = E0
        idx_spike.append(k + 1)
spikes = np.zeros((L,))
spikes[idx_spike] = 1

# Plot results
plt.figure(figsize=(11, 8))
plt.plot(t, I, 'k')
plt.xlabel('Time (ms)')
plt.ylabel('Input Current (nA)')
plt.title('Input Current')

plt.figure(figsize=(11, 8))
plt.plot(t, V, 'k')
plt.plot(t, Vt * np.ones(L,), 'r--')
plt.legend(['Membrane Potential', 'Threshold'])
plt.ylabel('Membrane Potential (mV)')
plt.xlabel('Time (ms)')
plt.title('Membrane Potential')

plt.figure(figsize=(11, 8))
plt.plot(t, spikes, 'k')
plt.xlabel('Time (ms)')
plt.title('Spikes')
plt.show()
