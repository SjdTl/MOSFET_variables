# Libraries
import matplotlib.pyplot as plt
import numpy as np
import os


# Variables that determine if plots should be plotted
plot_square_root = False
plot_derivative = False
plot_double_derivative = False
plot_tangent = False
plot_threshold = False
plot_k = False
plot_Id_to_VDS = False
plot_derivative_VDS = False
plot_linear_k = False
plot_tangent_VDS = False
plot_lambda = True

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

if plot_square_root == True:
    # Plot
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("Square root of drain current against gate voltage for different values of $V_{DS}$")

    plt.ylabel("$\sqrt{I_D}$ ($\sqrt{A}$)")
    plt.xlabel("$V_{GS}$ (V)")
    plt.legend()
    plt.savefig(fname="Gate_voltage_square_current",dpi=1000)

# Differentiating
for i in range(0, amount_runs):
    I_D_sqrt_diff[i] = np.gradient(I_D_sqrt[i], V_GS[i])

if plot_derivative == True:
    # Plot derivative
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'derivative $V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("$\sqrt{I_D}$ against $V_{GS}$ for different values of $V_{DS}$ with derivatives")

    plt.ylabel("$\sqrt{I_D}$ ($\sqrt{A}$)")
    plt.xlabel("$V_{GS}$ (V)")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname="Gate_voltage_square_current_derivatives", dpi=1000)

# Differentiating
for i in range(0, amount_runs):
    I_D_sqrt_diff2[i] = np.gradient(I_D_sqrt_diff[i], V_GS[i])

if plot_double_derivative == True:
    # Plot derivative
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'derivative $V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i], I_D_sqrt_diff2[i], label = 'Double derivative $V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("Test")

    plt.legend()

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

a = []
b = []
# Determine slope at inflection point
for i in range(0, amount_runs):
    a.append(I_D_sqrt_diff[i][intersection_voltage_element[i]])
    # a(V_GS)+b=I_D_sqrt so b=I_D_sqrt-(a*V_GS) 
    b.append(I_D_sqrt[i][intersection_voltage_element[i]]-a[i]*intersection_voltage[i])

def tangent_VGS(V_GS, i):
    # Only np arrays can multiply element-wise
    return (a[i])*np.array(V_GS)+b[i]

if plot_tangent == True:
    # Plot
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i],tangent_VGS(V_GS[i], i), label = 'Tangent of $V_{DS}=$' + str(V_DS[i]) + '$V$')
    
    ax.set_title("Extrapolation of the straight-line segments")
    plt.ylabel("$\sqrt{I_D}$ ($\sqrt{A}$)")
    plt.xlabel("$V_{GS}$ (V)")
    plt.grid(linewidth=0.1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname="tangents_VDS", dpi=1000)

# To determine when intersecting with x-axis, the following holds: aV_GS+b=0
# Therefore $V_{T}$ is given by $V_{T}=-b/a
V_T=[]
for i in range(0, amount_runs):
    V_T.append(-b[i]/a[i])


# Degree is one, since there is a linear connection (the modulation is given by ($1-\lambda V_{DS}$) )
y=np.polynomial.polynomial.Polynomial.fit(V_DS,V_T,1)
x = np.linspace(0,2,100)
y_values = np.polyval(np.flip(y.convert().coef), x)

# Plot a and V_T vs. V_DS
if plot_threshold:
    fig, ax = plt.subplots()
    ax.plot(x, y_values, label="Linearized values")
    ax.plot(V_DS, V_T, label="Experimental values")
    ax.scatter(0,y_values[0])
    ax.annotate("(0,"+str(round(y_values[0],3))+")", (0,y_values[0]))

    plt.title("Threshold voltages")
    plt.xlabel("$V_{DS}$ (V)")
    plt.ylabel("$V_{T}$ (V)")
    plt.legend()
    plt.savefig(fname="Threshold_voltages", dpi=1000)

determined_VT=y_values[0]

# DETERMINE K FROM SATURATION REGION
# The derivative at the saturation region is already determined. This is stored in the matrix a
# k = 2(I_{D}'^2) = 2(a^2)
saturation_k=2*np.square(a)*1000

# plot these values with a linearization
y=np.polynomial.polynomial.Polynomial.fit(V_DS,saturation_k,1)
x = np.linspace(0,2,100)
y_values = np.polyval(np.flip(y.convert().coef), x)

# Plot k vs V_DS
if plot_k:
    fig, ax = plt.subplots()
    ax.plot(V_DS, saturation_k, label="Experimental values")
    ax.plot(x, y_values, label="Linearized values", linewidth=1)

    ax.scatter(0,y_values[0])
    ax.annotate("(0,"+str(round(y_values[0],4))+"m)", (0,y_values[0]))

    plt.title("Values of $k$ for different drain voltages")
    plt.xlabel("$V_{DS}$ (V)")
    plt.ylabel("$k$ ($mA/V^{2}$)")
    plt.legend()
    plt.savefig(fname="k_values", dpi=1000)


# I_D TO VDS
filename="VDS calculations big.txt"
filepath = os.path.join(folder, filename)

# Find the amount of runs
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        
        # Form of the step data: "Step Information: Vgs=0  (Run: 1/7)"
        if ("Step Information:" in line):
            # Split up the string in an array of three parts using partition at the value .partition("value") and take appropriate index
            # So first partition gives "Step Information: Vds=0  (Run: 1/7)".partition("Run:") --> ["Step Information: Vds=0  (", "Run:", " 1/7)"] [2] --> " 1/7" 
            amount_runs = int(((line.partition("(Run:")[2]).partition("/")[2]).partition(")")[0])
            break
# Declare arrays
data = [[] for _ in range(amount_runs)]
V_DS = [[] for _ in range(amount_runs)]
I_D = [[] for _ in range(amount_runs)]
I_D_sqrt = [[] for _ in range(amount_runs)]

I_D_diff = [[] for _ in range(amount_runs)]
I_D_diff2 = [[] for _ in range(amount_runs)] #maybe not necessary

temp = ""
V_GS = []
run = -1
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if ("Step Information:" in line):
            run += 1
            temp = str((line.partition("Vgs=")[2]).partition(" ")[0])
            lastLetter = temp[len(temp)-1]
            if (lastLetter == "m") :
                temp = float(temp.partition("m")[0]) * 0.001
            V_GS.append(float(temp))
        else:
            if (run != -1):
                data[run].append((line.strip("\n")).split("\t"))
run_amount=0
for run in data:
    for elements in run:
        V_DS[run_amount].append(float(elements[0]))
        I_D[run_amount].append(float(elements[1]))
        I_D_sqrt[run_amount].append(float(elements[2]))
    run_amount += 1

if plot_Id_to_VDS == True:
    # Plot
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        if (i%4==1):
            ax.plot(V_DS[i], np.multiply(I_D[i],1000), label = '$V_{GS}=$' + str(round(V_GS[i],1)) + '$V$')
    ax.set_title("Drain current against drain source voltage for different values of $V_{GS}$")

    plt.ylabel("${I_D}$ (${mA}$)")
    plt.xlabel("$V_{DS}$ (V)")
    plt.legend()
    plt.savefig(fname="Drain_voltage_current",dpi=1000)

# Differentiating
for i in range(0, amount_runs):
    I_D_diff[i] = np.gradient(I_D[i], V_DS[i])

if plot_derivative_VDS == True:
    # Plot derivative
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        if (i%4==1):
            ax.plot(V_DS[i], np.multiply(I_D[i],1000), label = '$V_{GS}=$' + str(round(V_GS[i],1)) + '$V$')
            ax.plot(V_DS[i], np.multiply(I_D_diff[i],1000), label = 'Derivative $V_{GS}=$' + str(round(V_GS[i],1)) + '$V$')
    ax.set_title("$I_D$ against $V_{DS}$ for different values of $V_{GS}$ with derivatives")

    plt.ylabel("${I_D}$ (${mA}$)")
    plt.xlabel("$V_{DS}$ (V)")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height*0.5])
    plt.tight_layout(rect=[0,0,0.70,0.95])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname="Drain_voltage_square_current_derivatives", dpi=1000)

ID_derivatives_at_zero = [] 
linear_k = []
for i in range(0, amount_runs):
    ID_derivatives_at_zero.append(I_D_diff[i][0])
    linear_k.append(ID_derivatives_at_zero[i]/(V_GS[i]-determined_VT))

# Plot k vs V_GS
if plot_linear_k:
    fig, ax = plt.subplots()
    ax.plot(V_GS, np.multiply(linear_k,1000), label="Experimental values")

    plt.title("Values of $k$ for different gate voltages")
    plt.xlabel("$V_{GS}$ (V)")
    plt.ylabel("$k$ ($mA/V^{2}$)")
    plt.savefig(fname="linear_k_values", dpi=1000)

intersection_voltage = []
intersection_voltage_element = []

a = []
b = []
# Calculate intersection voltages: ax+b=0 --> x=-b/a
intersection_voltage_lambda = []
lambda_value = []

for i in range (0, amount_runs):
    for j in range (0,len(V_DS[i])-1):
        if (V_DS[i][j] > 2):
            a.append(I_D_diff[i][j])
            b.append(I_D[i][j]-a[i]*V_DS[i][j])
            intersection_voltage_lambda.append(-b[i]/a[i])
            lambda_value.append(-1/intersection_voltage_lambda[i])
            break

def tangent_VDS(V_DS, i):
    # Only np arrays can multiply element-wise
    return (a[i])*np.array(V_DS)+b[i]

x = np.linspace(-10,2,100)

if plot_tangent_VDS == True:
    # Plot
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        # Do not plot all values
        if (i%4==1):
            ax.plot(V_DS[i], np.multiply(I_D[i],1000), label = '$V_{GS}=$' + str(V_GS[i]) + '$V$')
            ax.plot(x,np.multiply(tangent_VDS(x, i),1000), label = 'Tangent of $V_{GS}=$' + str(V_GS[i]) + '$V$')
    
    ax.set_title("Extrapolation of the straight-line segments")
    plt.ylabel("${I_D}$ ($m{A}$)")
    plt.xlabel("$V_{DS}$ (V)")
    plt.grid(linewidth=0.1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname="tangents_VDS", dpi=1000)


# Degree is one, since there is a linear connection (the modulation is given by ($1-\lambda V_{DS}$) )
y=np.polynomial.polynomial.Polynomial.fit(V_GS,lambda_value,1)
x = np.linspace(0,2,100)
y_values = np.polyval(np.flip(y.convert().coef), x)

# Plot a and V_T vs. V_DS
if plot_lambda:
    fig, ax = plt.subplots()
    ax.plot(x, y_values, label="Linearized values")
    ax.plot(V_GS, lambda_value, label="Experimental values")
    ax.scatter(0,y_values[0])
    ax.annotate("(0,"+str(round(y_values[0],3))+")", (0,y_values[0]))

    plt.title("Value for $\lambda$")
    plt.xlabel("$V_{GS}$ (V)")
    plt.ylabel("$\lambda$ ($V^{-1}$)")
    plt.legend()
    plt.savefig(fname="Lambda_values", dpi=1000)