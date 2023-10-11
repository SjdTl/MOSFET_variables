# Libraries
import matplotlib.pyplot as plt
import numpy as np
import os


# Variables that determine if plots should be plotted
plot_Id_to_Vgs = True
plot_rootId_to_Vgs = True
plot_rootId_to_Vgs_derivative = True
plot_rootId_to_Vgs_derivative2 = True
plot_rootId_to_Vgs_tangent = True
plot_Vt_to_Vds = True
plot_k_to_Vds = True
plot_Id_to_Vds = True
plot_Id_to_Vds_derivative = True
plot_k_to_Vgs = True
plot_Id_to_Vds_tangent = True 
plot_Lambda_to_Vgs = True

# Filename of both LTSpice outputs
filename_Id_to_Vgs = "VGS calculations.txt"
filename_Id_to_Vds = "VDS calculations big.txt"
thisFolder = os.path.dirname(os.path.abspath(__file__))

filepath=os.path.join(thisFolder, filename_Id_to_Vgs)

# Take SI prefix and calculate the base 0 value
def remove_SI_prefix(value):
    lastLetter = value[len(value)-1]
    if (lastLetter == "m") :
        value = float(value.partition("m")[0]) * 0.001
    return value

def tangent(V, i):
    # Only np arrays can multiply element-wise
    return (a[i])*np.array(V)+b[i]

# Find the amount of runs
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Form of the step data: "Step Information: Vds=200m  (Run: 1/7)"
        if ("Step Information:" in line):
            # Split up the string in an array of three parts using partition
            # So first partition gives "Step Information: Vds=0  (Run: 1/7)".partition("Run:") 
            # --> ["Step Information: Vds=0  (", "Run:", " 1/7)"] [2] 
            # --> " 1/7" 
            amount_runs = int(((line.partition("(Run:")[2]).partition("/")[2]).partition(")")[0])
            break

# Declare arrays
data = [[] for _ in range(amount_runs)]
V_GS = [[] for _ in range(amount_runs)]
I_D = [[] for _ in range(amount_runs)]
I_D_sqrt = [[] for _ in range(amount_runs)]

# Stores step information per run
V_DS = []

I_D_sqrt_diff = [[] for _ in range(amount_runs)]
I_D_sqrt_diff2 = [[] for _ in range(amount_runs)]

# Read out file and store in data array
run = -1
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if ("Step Information:" in line):
            run += 1
            Vds_with_prefix = str((line.partition("Vds=")[2]).partition(" ")[0])
            V_DS.append(float(remove_SI_prefix(Vds_with_prefix)))
        else:
            if (run != -1):
                data[run].append((line.strip("\n")).split("\t"))

# Split up data array into the Vgs, Id and sqrt(Id) array
run_amount=0
for run in data:
    for elements in run:
        V_GS[run_amount].append(float(elements[0]))
        I_D[run_amount].append(float(elements[1]))
        I_D_sqrt[run_amount].append(float(elements[2]))
    run_amount += 1

if plot_Id_to_Vgs == True:
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], np.multiply(I_D[i],1000), label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("Drain current to gate voltage for different values of $V_{DS}$")

    plt.ylabel("${I_D}$ ($m{A}$)")
    plt.xlabel("$V_{GS}$ (V)")
    plt.legend()
    plt.savefig(fname="Id_to_Vgs_plot",dpi=1000)

if plot_rootId_to_Vgs == True:
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("Square root of drain current to gate voltage for different values of $V_{DS}$")

    plt.ylabel("$\sqrt{I_D}$ ($\sqrt{A}$)")
    plt.xlabel("$V_{GS}$ (V)")
    plt.legend()
    plt.savefig(fname="sqrt(Id)_to_Vgs_plot",dpi=1000)

# Differentiating
for i in range(0, amount_runs):
    I_D_sqrt_diff[i] = np.gradient(I_D_sqrt[i], V_GS[i])

if plot_rootId_to_Vgs_derivative == True:
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i], I_D_sqrt_diff[i], label = 'derivative $V_{DS}=$' + str(V_DS[i]) + '$V$')
    ax.set_title("$\sqrt{I_D}$ to $V_{GS}$ for different values of $V_{DS}$ with derivatives")

    plt.ylabel("$\sqrt{I_D}$ ($\sqrt{A}$)")
    plt.xlabel("$V_{GS}$ (V)")

    # Put a legend to the right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname="rootId_to_Vgs_derivative", dpi=1000)

# Differentiating
for i in range(0, amount_runs):
    I_D_sqrt_diff2[i] = np.gradient(I_D_sqrt_diff[i], V_GS[i])

# Determine zeros
intersection_voltage = []
intersection_voltage_element = []

for i in range (0, amount_runs):
    intersection_voltage.append(0)
    intersection_voltage_element.append(0)
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

if plot_rootId_to_Vgs_tangent == True:
    # Plot
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        ax.plot(V_GS[i], I_D_sqrt[i], label = '$V_{DS}=$' + str(V_DS[i]) + '$V$')
        ax.plot(V_GS[i],tangent(V_GS[i], i), label = 'Tangent of $V_{DS}=$' + str(V_DS[i]) + '$V$')
    
    ax.set_title("Extrapolation of the straight-line segments")
    plt.ylabel("$\sqrt{I_D}$ ($\sqrt{A}$)")
    plt.xlabel("$V_{GS}$ (V)")
    plt.grid(linewidth=0.1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname="rootId_to_Vgs_tangent", dpi=1000)

# To determine when intersecting with x-axis, the following holds: aV_GS+b=0
# Therefore $V_{T}$ is given by $V_{T}=-b/a
V_T=[]
for i in range(0, amount_runs):
    V_T.append(-b[i]/a[i])


# Degree is one, since there is a linear connection (the modulation is given by ($1-\lambda V_{DS}$) )
y=np.polynomial.polynomial.Polynomial.fit(V_DS,V_T,1)
x=np.linspace(0,2,100)
y_values = np.polyval(np.flip(y.convert().coef), x)

# Plot a and V_T vs. V_DS
if plot_Vt_to_Vds:
    fig, ax = plt.subplots()
    ax.plot(x, y_values, label="Linearized values")
    ax.plot(V_DS, V_T, label="Experimental values")
    ax.scatter(0,y_values[0])
    ax.annotate("(0,"+str(round(y_values[0],3))+")", (0,y_values[0]))

    plt.title("Threshold voltages")
    plt.xlabel("$V_{DS}$ (V)")
    plt.ylabel("$V_{T}$ (V)")
    plt.legend()
    plt.savefig(fname="Vt_to_Vds", dpi=1000)

determined_VT=y_values[0]
print("Vt0=" + str(determined_VT))

# DETERMINE K FROM SATURATION REGION
# The derivative at the saturation region is already determined. This is stored in the matrix a
# k = 2(I_{D}'^2) = 2(a^2)
saturation_k=2*np.square(a)*1000

# plot these values with a linearization
y=np.polynomial.polynomial.Polynomial.fit(V_DS,saturation_k,1)
x = np.linspace(0,2,100)
y_values = np.polyval(np.flip(y.convert().coef), x)

# Plot k vs V_DS
if plot_k_to_Vds:
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

determined_k_Vds = y_values[0]
print("k=" + str(determined_k_Vds) + "m (from Id against Vgs)")



#------------------------------------------------------
# I_D TO VDS
filepath = os.path.join(thisFolder, filename_Id_to_Vds)

# Find the amount of runs
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if ("Step Information:" in line):
            amount_runs = int(((line.partition("(Run:")[2]).partition("/")[2]).partition(")")[0])
            break

# Declare (or redeclare) arrays
data = [[] for _ in range(amount_runs)]
V_DS = [[] for _ in range(amount_runs)]
I_D = [[] for _ in range(amount_runs)]
I_D_diff = [[] for _ in range(amount_runs)]
V_GS = [] #empty Vgs

run = -1
with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if ("Step Information:" in line):
            run += 1
            Vgs_with_prefix = str((line.partition("Vgs=")[2]).partition(" ")[0])
            V_GS.append(float(remove_SI_prefix(Vgs_with_prefix)))
        else:
            if (run != -1):
                data[run].append((line.strip("\n")).split("\t"))
run_amount=0
for run in data:
    for elements in run:
        V_DS[run_amount].append(float(elements[0]))
        I_D[run_amount].append(float(elements[1]))
    run_amount += 1

if plot_Id_to_Vds == True:
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        if (i%4==1):
            ax.plot(V_DS[i], np.multiply(I_D[i],1000), label = '$V_{GS}=$' + str(round(V_GS[i],1)) + '$V$')
    ax.set_title("Drain current to drain source voltage for different values of $V_{GS}$")

    plt.ylabel("${I_D}$ (${mA}$)")
    plt.xlabel("$V_{DS}$ (V)")
    plt.legend()
    plt.savefig(fname="Id_to_Vds",dpi=1000)

# Differentiating
for i in range(0, amount_runs):
    I_D_diff[i] = np.gradient(I_D[i], V_DS[i])

if plot_Id_to_Vds_derivative == True:
    # Plot derivative
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        # Not everything is plotted, because a lot of runs are used
        if (i%4==1):
            ax.plot(V_DS[i], np.multiply(I_D[i],1000), label = '$V_{GS}=$' + str(round(V_GS[i],1)) + '$V$')
            ax.plot(V_DS[i], np.multiply(I_D_diff[i],1000), label = 'Derivative $V_{GS}=$' + str(round(V_GS[i],1)) + '$V$')
    ax.set_title("$I_D$ to $V_{DS}$ for different values of $V_{GS}$ with derivatives")

    plt.ylabel("${I_D}$ (${mA}$)")
    plt.xlabel("$V_{DS}$ (V)")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height*0.5])
    plt.tight_layout(rect=[0,0,0.70,0.95])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname="Id_to_Vds_derivative", dpi=1000)

ID_derivatives_at_zero = [] 
linear_k = []
for i in range(0, amount_runs):
    ID_derivatives_at_zero.append(I_D_diff[i][0])
    linear_k.append(ID_derivatives_at_zero[i]/(V_GS[i]-determined_VT))

average_k=0
total=0
amount=0
for i in range(0,amount_runs):
    if (V_GS[i]>0.6 and V_GS[i] < 1.8):
        total =+ linear_k[i]
        amount =+ 1
average_k = total/amount
print("k="+str(average_k*1000) + "m (from Id against Vds)")


# Plot k vs V_GS
if plot_k_to_Vgs:
    fig, ax = plt.subplots()
    ax.plot(V_GS, np.multiply(linear_k,1000), label="Experimental values")

    plt.title("Values of $k$ for different gate voltages")
    plt.xlabel("$V_{GS}$ (V)")
    plt.ylabel("$k$ ($mA/V^{2}$)")
    plt.savefig(fname="k_to_Vgs", dpi=1000)

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

x = np.linspace(-10,2.5,100)
if plot_Id_to_Vds_tangent == True:
    # Plot
    fig, ax = plt.subplots()
    for i in range(0, amount_runs):
        # Do not plot all values
        if (i%4==1):
            ax.plot(V_DS[i], np.multiply(I_D[i],1000), label = '$V_{GS}=$' + str(V_GS[i]) + '$V$')
            ax.plot(x,np.multiply(tangent(x, i),1000), label = 'Tangent of $V_{GS}=$' + str(V_GS[i]) + '$V$')
    
    ax.set_title("Extrapolation of the straight-line segments")
    plt.ylabel("${I_D}$ ($m{A}$)")
    plt.xlabel("$V_{DS}$ (V)")
    plt.grid(linewidth=0.1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    ax.set_ylim(-0.2,2.3)
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname="Id_to_Vds_tangent", dpi=1000)

total=0
amount=0
for i in range(0,amount_runs):
    if (V_GS[i]>0.6 and V_GS[i] < 1.8):
        total =+ lambda_value[i]
        amount =+ 1
average_lambda = total/amount
print("lambda=" + str(average_lambda))
        
if plot_Lambda_to_Vgs:
    fig, ax = plt.subplots()
    ax.plot(V_GS,lambda_value, label="Experimental values")

    ax.set_xlim(0.75,max(V_GS))
    ax.set_ylim(0,0.4)
    plt.title("Value for $\lambda$")
    plt.xlabel("$V_{GS}$ (V)")
    plt.ylabel("$\lambda$ ($V^{-1}$)")
    plt.savefig(fname="Lambda_values", dpi=1000)