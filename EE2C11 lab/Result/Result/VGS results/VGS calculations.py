# TO DO
# - Automate the reading of V_DS
# - Deal with V_DS=0, such that all runs can be used and the code is easier to read
# - Change names

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import os


# Variables that determine if plots should be plotted
plot = True
plot_differential = True
plot_double_differential = False
plot_tangent = True
plot_threshold = True

# Find filename and location
filename = "VGS calculations.txt"
folder = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(folder, filename)

# Declare variables 
amount_runs = -1

# Find the amount of runs
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        
        # Form of the step data: "Step Information: Vds=0  (Run: 1/7)"
        if ("Step Information:" in line):
            # Split up the string in an array of three parts using partition at the value .partition("value") and take appropriate index
            # So first partition gives "Step Information: Vds=0  (Run: 1/7)".partition("Run:") --> ["Step Information: Vds=0  (", "Run:", " 1/7)"] [2] --> " 1/7" 
            amount_runs = int(((line.partition("(Run:")[2]).partition("/")[2]).partition(")")[0])
            break

print(amount_runs)

# Declare arrays
data = [[] for _ in range(amount_runs)]
V_GS = [[] for _ in range(amount_runs)]
I_D = [[] for _ in range(amount_runs)]
I_D_sqrt = [[] for _ in range(amount_runs)]

I_D_sqrt_diff = [[] for _ in range(amount_runs)]
I_D_sqrt_diff2 = [[] for _ in range(amount_runs)]

temp = ""
V_DS = []
run = -1
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if ("Step Information:" in line):
            run += 1
            temp = str((line.partition("Vds=")[2]).partition(" ")[0])
            lastLetter = temp[len(temp)-1]
            if (lastLetter == "m") :
                temp = float(temp.partition("m")[0]) * 0.001
            V_DS.append(float(temp))
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
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("Test")

    plt.legend()
    plt.show()

# Differentiating
for i in range(0, amount_runs):
    I_D_sqrt_diff[i] = np.gradient(I_D_sqrt[i], V_GS[i])

if plot_differential == True:
    # Plot differential
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'Differential $V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("Test")

    plt.legend()
    plt.show()

# Differentiating
for i in range(0, amount_runs):
    I_D_sqrt_diff2[i] = np.gradient(I_D_sqrt_diff[i], V_GS[i])

if plot_double_differential == True:
    # Plot differential
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'Differential $V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i], I_D_sqrt_diff2[i], label = 'Double differential $V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("Test")

    plt.legend()
    plt.show()

# Determine zeros
#Start array with 1 element, since the first run is skipped
intersection_voltage = []
intersection_voltage_element = []

for i in range (0, amount_runs):
    intersection_voltage.append(9999)
    intersection_voltage_element.append(-1)
    lowest_value = 9999
    for j in range (0,len(I_D_sqrt_diff2[i])-1):
        if lowest_value > abs(I_D_sqrt_diff2[i][j]):
            lowest_value = abs(I_D_sqrt_diff2[i][j])
            intersection_voltage[i]=(V_GS[i][j])
            intersection_voltage_element[i]=j
print(intersection_voltage)

a = []
b = []
# Determine slope at inflection point
for i in range(0, amount_runs):
    a.append(I_D_sqrt_diff[i][intersection_voltage_element[i]])
    # a(V_GS)+b=I_D_sqrt so b=I_D_sqrt-(a*V_GS) 
    b.append(I_D_sqrt[i][intersection_voltage_element[i]]-a[i]*intersection_voltage[i])

print(a)
print(b)

def tangent(V_GS, i):
    # Only np arrays can multiply element-wise
    return (a[i])*np.array(V_GS)+b[i]

if plot_tangent == True:
    # Plot
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i],tangent(V_GS[i], i), label = 'Tangent of $V_{DS}=$' + str(V_DS[i]) + '$V$')
    
    ax.set_title("Extrapolation of the straight-line segments")
    plt.xlabel("$V_{DS}$ (V)")
    plt.ylabel("$V_T$ (V)")
    plt.legend()
    plt.grid(linewidth=0.1)
    plt.show()

# To determine when intersecting with x-axis, the following holds: aV_GS+b=0
# Therefore $V_{T}$ is given by $V_{T}=-b/a
V_T=[]
for i in range(0, amount_runs):
    V_T.append(-b[i]/a[i])

print(V_T)


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

    plt.title("Threshold values")
    plt.xlabel("$V_{DS}$ (V)")
    plt.ylabel("$V_{T}$ (V)")
    plt.legend()
    plt.show()


# When $V_DS=0$, there is no modulation and $V_T$ is most accurate. $V_DS=0$ is given by y_values[0] (see prev code)
print(y_values[0])