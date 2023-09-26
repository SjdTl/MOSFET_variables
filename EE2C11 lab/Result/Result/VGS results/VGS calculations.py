# Libraries
import matplotlib.pyplot as plt
import numpy as np
import os

amount_runs = 7
amount_to_plot = 7

data = [[] for _ in range(amount_runs)]
V_GS = [[] for _ in range(amount_runs)]
I_D = [[] for _ in range(amount_runs)]
I_D_sqrt = [[] for _ in range(amount_runs)]

I_D_sqrt_diff = [[] for _ in range(amount_runs)]
I_D_sqrt_diff2 = [[] for _ in range(amount_runs)]

filename = "VGS calculations.txt"
folder = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(folder, filename)

run = -1
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if ("Step Information:" in line):
            run += 1
        else:
            if (run != -1):
                data[run].append((line.strip("\n")).split("\t"))

run_amount=0
for run in data:
    for elements in run:
        V_GS[run_amount].append(float(elements[0]))
        I_D[run_amount].append(float(elements[1]))
        I_D_sqrt[run_amount].append(float(elements[2]))
    run_amount += 1

# Plot
fig, ax = plt.subplots()
for i in range(0, amount_to_plot):
    ax.plot(V_GS[i], I_D_sqrt[i], label = 'Run ' + str(i+1))
ax.set_title("Test")

plt.legend()
plt.show()

# Differentiating
for i in range(0, amount_to_plot):
    I_D_sqrt_diff[i] = np.gradient(I_D_sqrt[i], V_GS[i])

# Plot differential
fig, ax = plt.subplots()
for i in range(0, amount_to_plot):
    ax.plot(V_GS[i], I_D_sqrt[i], label = 'Run ' + str(i+1))
    ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'Differential run ' + str(i+1))
ax.set_title("Test")

plt.legend()
plt.show()

# Differentiating
for i in range(0, amount_to_plot):
    I_D_sqrt_diff2[i] = np.gradient(I_D_sqrt_diff[i], V_GS[i])

# Plot differential
fig, ax = plt.subplots()
for i in range(0, amount_to_plot):
    ax.plot(V_GS[i], I_D_sqrt[i], label = 'Run ' + str(i+1))
    ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'Differential run ' + str(i+1))
    ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'Differential run ' + str(i+1))
ax.set_title("Test")

plt.legend()
plt.show()