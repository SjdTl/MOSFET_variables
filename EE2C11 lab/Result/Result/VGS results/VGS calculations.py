# TO DO
# - Automate the reading of V_DS
# - Deal with V_DS=0, such that all runs can be used and the code is easier to read

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import os

plot = False
plot_differential = False
plot_double_differential = False
plot_tangent = False
plot_threshold = True

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

if plot == True:
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

if plot_differential == True:
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

if plot_double_differential == True:
    # Plot differential
    fig, ax = plt.subplots()
    for i in range(0, amount_to_plot):
        ax.plot(V_GS[i], I_D_sqrt[i], label = 'Run ' + str(i+1))
        ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'Differential run ' + str(i+1))
        ax.plot(V_GS[i], I_D_sqrt_diff2[i], label = 'Double differential run ' + str(i+1))
    ax.set_title("Test")

    plt.legend()
    plt.show()

# Determine zeros
#Start array with 1 element, since the first run is skipped
intersection_voltage = []
intersection_voltage_element = []

for i in range (1, amount_runs):
    intersection_voltage.append(9999)
    intersection_voltage_element.append(-1)
    lowest_value = 9999
    for j in range (0,len(I_D_sqrt_diff2[i])-1):
        if lowest_value > abs(I_D_sqrt_diff2[i][j]):
            lowest_value = abs(I_D_sqrt_diff2[i][j])
            intersection_voltage[i-1]=(V_GS[i][j])
            intersection_voltage_element[i-1]=j
print(intersection_voltage)

a = []
b = []
# Determine slope at inflection point
for i in range(1, amount_runs):
    a.append(I_D_sqrt_diff[i][intersection_voltage_element[i-1]])
    # a(V_GS)+b=I_D_sqrt so b=I_D_sqrt-(a*V_GS) 
    b.append(I_D_sqrt[i][intersection_voltage_element[i-1]]-a[i-1]*intersection_voltage[i-1])

print(a)
print(b)

def tangent(V_GS, i):
    # Only np arrays can multiply element-wise
    return (a[i])*np.array(V_GS)+b[i]

if plot_tangent == True:
    # Plot
    fig, ax = plt.subplots()
    for i in range(1, amount_to_plot):
        ax.plot(V_GS[i], I_D_sqrt[i], label = 'Run ' + str(i+1))
        ax.plot(V_GS[i],tangent(V_GS[i], i-1), label = 'Tangent ' + str(i+1))
    
    ax.set_title("Test")

    plt.legend()
    plt.show()

# To determine when intersecting with x-axis, the following holds: aV_GS+b=0
# Therefore $V_{T}$ is given by $V_{T}=-b/a
V_T=[]
for i in range(1, amount_runs):
    V_T.append(-b[i-1]/a[i-1])

print(V_T)

# Automate this by reading from the runs:
V_DS = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]


# Degree is one, since there is a linear connection (the modulation is given by ($1-\lambda V_{DS}$) )
y=np.polynomial.polynomial.Polynomial.fit(V_DS,V_T,1)
x = np.linspace(0,2,100)
y_values = np.polyval(np.flip(y.convert().coef), x)

# Plot a and V_T vs. V_DS
if plot_threshold:
    fig, ax = plt.subplots()
    ax.plot(x, y_values, label="$a$ for $V_{DS}$")
    ax.plot(V_DS, V_T, label="$V_T$ to $V_{DS}$")
    ax.scatter(0,y_values[0])
    ax.annotate("(0,"+str(round(y_values[0],2))+")", (0,y_values[0]))

    plt.xlabel("$V_{DS}$")
    plt.ylabel("Values")
    plt.legend()
    plt.show()


# When $V_DS=0$, there is no modulation and $V_T$ is most accurate. $V_DS=0$ is given by y_values[0] (see prev code)
print(y_values[0])