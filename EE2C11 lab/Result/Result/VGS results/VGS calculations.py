import matplotlib.pyplot as plt
import numpy as np
import os

runs = 7

data = [[] for _ in range(runs)]
V_GS = [[] for _ in range(runs)]
I_D = [[] for _ in range(runs)]
I_D_sqrt = [[] for _ in range(runs)]

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
#test

# Plot
fig, ax = plt.subplots()
for i in range(0, runs):
    ax.plot(V_GS[i], I_D_sqrt[i], label = 'Run ' + str(i+1))
ax.set_title("Test")

plt.legend()
plt.show()