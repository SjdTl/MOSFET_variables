import matplotlib.pyplot as plt
import numpy as np
import os

# VALUES
VT0=0.338
k = 2.2298*0.001
Lambda = 0.169
Vdsat = 0.39

def unified(VT0, k, Lambda, Vdsat, VDS, VGS):
    # RETURNS VALUE IN mA!
    Id=[]
    VGT=VGS-VT0
    for i in range(len(VDS)):
        Vmin=min(VGT,VDS[i],Vdsat)
        modulation=1+Lambda*VDS[i]
        Id.append(1000*k*(VGT*Vmin-0.5*Vmin*Vmin)*modulation)
    return Id


folder = os.path.dirname(os.path.abspath(__file__))
filename="VDS calculations.txt"
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

color = ["blue", "orange", "green", "red", "purple"]

# Plot
fig, ax = plt.subplots()
for i in range(0, amount_runs):
    ax.plot(V_DS[i], np.multiply(I_D[i],1000), label = '$V_{GS}=$' + str(round(V_GS[i],1)) + '$V$', color = color[i])
    ax.plot(V_DS[i], unified(VT0,k,Lambda,Vdsat,V_DS[i],V_GS[i]), label = 'Model $V_{GS}=$' + str(round(V_GS[i],1)) + '$V$', linestyle='dashed', color = color[i])
ax.set_title("Model and simulation for an NMOS transistor")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.ylabel("${I_D}$ (${mA}$)")
plt.xlabel("$V_{DS}$ (V)")

# Put a legend to the right of the current axis
plt.savefig(fname="Unified model",dpi=1000)
