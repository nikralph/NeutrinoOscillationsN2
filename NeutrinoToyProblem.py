import numpy as np
import matplotlib.pyplot as plt

from scipy.special import comb

# Time Evolution Operator; Returns probability
def timeywimey(vals, vecs, state, time):
    for i in range(len(vals)):
        vals[i] = np.exp(-1j*vals[i]*time)
    vals = np.diag(vals)
    expectation = state.T@(vecs@vals@np.linalg.inv(vecs))@state
    probability = expectation*np.conjugate(expectation)
    return probability[0,0]

# Graph data generator for time evolution operator expectation values
def timeyGraphdata(vals, vecs, state, time, resolution):
    times = np.zeros(resolution, dtype=complex)
    ExpVals = np.zeros(resolution, dtype=complex)
    for i in range(resolution):
        times[i] = (i*time)/resolution
        ExpVals[i] = timeywimey(np.copy(vals), vecs, state, times[i])
    return times, ExpVals

# Formatting
np.set_printoptions(formatter={'all': lambda x: "{:.12g}".format(x)})

Nnu = 2     # Number of Neutrinos
Nflav = 2   # Number of Flavor states
Nk = 4      # Number of Momentum states

# dimension of Hilbert space: (Number of states) choose (Number of Neutrinos)
dim = comb((Nk*Nflav), Nnu, exact=True)

# fancyE = G_F / (sqrt(2)*V), set to 1 for simplicity.
fancyE = 1

# Define constants. Can expand on this with relative ease.
Tbar = 10**(4)
omega = 1
theta = (1/2)*np.arcsin(0.8)
r = 2.0
phi = np.pi/4
epsilons = [0,10**(-10/2),10**(-8/2),10**(-6/2),10**(-4/2),666,10**(-2/2)]
#epsilon = 0

M = np.array([[4,0,0,0],[0,2,2,0],[0,2,2,0],[0,0,0,4]], dtype=complex)

for epsilon in epsilons:
    # P~ vectors as row vectors; I later define Pbar as the norm of P~ vectors.
    P = np.array([[np.sin(phi),np.cos(phi),0],[np.sqrt(r**(2)-np.cos(phi)**(2)), -np.cos(phi),0],[np.sin(phi),-np.cos(phi)-epsilon,0],[np.sqrt(r**(2)-np.cos(phi)**(2)), np.cos(phi)+epsilon,0]], dtype=complex)

    # Initialize Hamiltonians
    H = np.zeros((dim, dim), dtype=complex)
    HT = np.zeros((dim, dim), dtype=complex)
    Hkin = np.zeros((dim, dim), dtype=complex)
    Hvv = np.zeros((dim, dim), dtype=complex)
    HvvT = np.zeros((dim, dim), dtype=complex)

    # Initialize variables
    Pbar = np.zeros(4, dtype=complex)
    Pnorm = np.zeros(4, dtype=complex)
    wbar = np.zeros(4, dtype=complex)
    fphi = np.zeros(4, dtype=complex)
    ftheta = np.zeros(4, dtype=complex)
    Fij = np.zeros((6,6), dtype=complex)

    # Define values based on chosen variables
    for i in range(len(P)):
        Pbar[i] = np.linalg.norm(P[i])
        # Pbar[i] = Pnorm[i] / T, Tbar = T / fancyE, simplifying to fancyE = 1
        Pnorm[i] = Pbar[i] * Tbar * fancyE
        wbar[i] = (omega * np.sin(2*theta)) / Pbar[i]
        fphi[i] = np.arctan(P[i][1] / P[i][0])
        
        # Define 4 1x1 blocks of Hamiltonian for when flav is the different but P is the same.
        H[i,i] = 2*Pnorm[i]
        HT[i,i] = 2*Pnorm[i]

    # Dealing with awkward, off-diagnal (F-dagger * F) terms
    Fij[4,5] = (np.sqrt(1/2) * (np.exp(1j*fphi[1-1]) - np.exp(1j*fphi[2-1]))) * (np.sqrt(1/2) * (np.exp(-1j*fphi[3-1]) - np.exp(-1j*fphi[4-1])))
    Fij[5,4] = (np.sqrt(1/2) * (np.exp(1j*fphi[3-1]) - np.exp(1j*fphi[4-1]))) * (np.sqrt(1/2) * (np.exp(-1j*fphi[1-1]) - np.exp(-1j*fphi[2-1])))

    Hvv[20:24,24:28] = Fij[4,5]*M
    Hvv[24:28,20:24] = Fij[5,4]*M

    # Build all 6 4x4 blocks for both Hkin and Hvv and put them into their relevent locations
    Dij = np.zeros((6,4), dtype=complex)
    for ij in range(len(Dij)):
        # Awkwardly set i and j for respective momentum modes
        if ij == 0:
            i = 1
            j = 2
        elif ij == 1:
            i = 1
            j = 3
        elif ij == 2:
            i = 1
            j = 4
        elif ij == 3:
            i = 2
            j = 3
        elif ij == 4:
            i = 2
            j = 4
        elif ij == 5:
            i = 3
            j = 4
        else:
            print("ERROR in Dij")
        
        # Fij represents (F-dagger * F); used for interaction term Hvv
        Fij[ij,ij] = (np.sqrt(1/2) * (np.exp(1j*fphi[i-1]) - np.exp(1j*fphi[j-1]))) * (np.sqrt(1/2) * (np.exp(-1j*fphi[i-1]) - np.exp(-1j*fphi[j-1])))
        
        Dij[ij,0] = (Tbar * (Pbar[i-1]+Pbar[j-1])) - ((omega * np.cos(2*theta))/Pbar[i-1]) - ((omega * np.cos(2*theta))/Pbar[j-1])
        Dij[ij,1] = (Tbar * (Pbar[i-1]+Pbar[j-1])) - ((omega * np.cos(2*theta))/Pbar[i-1]) + ((omega * np.cos(2*theta))/Pbar[j-1])
        Dij[ij,2] = (Tbar * (Pbar[i-1]+Pbar[j-1])) + ((omega * np.cos(2*theta))/Pbar[i-1]) - ((omega * np.cos(2*theta))/Pbar[j-1])
        Dij[ij,3] = (Tbar * (Pbar[i-1]+Pbar[j-1])) + ((omega * np.cos(2*theta))/Pbar[i-1]) + ((omega * np.cos(2*theta))/Pbar[j-1])
        
        # Define 4x4 kinetic block
        Hkin_temp = np.array([[Dij[ij,0],wbar[j-1],wbar[i-1],0],[wbar[j-1],Dij[ij,1],0,wbar[i-1]],[wbar[i-1],0,Dij[ij,2],wbar[j-1]],[0,wbar[i-1],wbar[j-1],Dij[ij,3]]], dtype=complex)
        
        # Write relevent 4x4s to kinetic and interaction terms
        if ij == 0:
            Hkin[(ij+5)*4:(ij+6)*4,(ij+5)*4:(ij+6)*4] = Hkin_temp
            Hvv[(ij+5)*4:(ij+6)*4,(ij+5)*4:(ij+6)*4] = Fij[ij,ij]*M
            HvvT[(ij+5)*4:(ij+6)*4,(ij+5)*4:(ij+6)*4] = Fij[ij,ij]*M
        elif ij == 5:
            Hkin[(ij+1)*4:(ij+2)*4,(ij+1)*4:(ij+2)*4] = Hkin_temp
            Hvv[(ij+1)*4:(ij+2)*4,(ij+1)*4:(ij+2)*4] = Fij[ij,ij]*M
            HvvT[(ij+1)*4:(ij+2)*4,(ij+1)*4:(ij+2)*4] = Fij[ij,ij]*M
        else:
            Hkin[(ij)*4:(ij+1)*4,(ij)*4:(ij+1)*4] = Hkin_temp
            Hvv[(ij)*4:(ij+1)*4,(ij)*4:(ij+1)*4] = Fij[ij,ij]*M
            HvvT[(ij)*4:(ij+1)*4,(ij)*4:(ij+1)*4] = Fij[ij,ij]*M

    # Define the Hamiltonian using kinetic and interaction terms
    H += (fancyE * (Hkin + Hvv))

    # Define the truncated version of the Hamiltonian
    HT += (fancyE * (Hkin + HvvT))

    # Diagonalize the Full 28x28 Hamiltonian
    Hval, Hvec = np.linalg.eigh(H)
    Hval = Hval.astype(dtype=complex)

    # Diagonalize the Truncated Hamiltonian
    HTval, HTvec = np.linalg.eigh(HT)
    HTval = HTval.astype(dtype=complex)

    # Define state |v1>
    State = np.zeros((28,1), dtype=complex)
    State[20] = 1

    # Choose time interval and resolution
    Timey = 10
    res = 1000

    if epsilon == 0:
        graphcolor = "#0000FF"
        graphline = "-"
        order = 500
        width = 2
    elif epsilon == 10**(-5):
        graphcolor = "#000000"
        graphline = "-"
        order = 1
        width = 1
    elif epsilon == 10**(-9/2):
        graphcolor = "#222222"
        graphline = "-"
        order = 2
        width = 1
    elif epsilon == 10**(-8/2):
        graphcolor = "#444444"
        graphline = "-"
        order = 3
        width = 1
    elif epsilon == 10**(-7/2):
        graphcolor = "#444444"
        graphline = "-"
        order = 4
        width = 1
    elif epsilon == 10**(-6/2):
        graphcolor = "#444444"
        graphline = "--"
        order = 5
        width = 1
    elif epsilon == 10**(-5/2):
        graphcolor = "#222200"
        graphline = "--"
        order = 6
        width = 1
    elif epsilon == 10**(-4/2):
        graphcolor = "#444400"
        graphline = "--"
        order = 7
        width = 1
    elif epsilon == 10**(-3/2):
        graphcolor = "#666600"
        graphline = "--"
        order = 8
        width = 1
    elif epsilon == 10**(-2/2):
        graphcolor = "#888800"
        graphline = "--"
        order = 9
        width = 1
    elif epsilon == 10**(-1/2):
        graphcolor = "#AAAA00"
        graphline = "--"
        order = 10
        width = 1
    elif epsilon == 666:
        graphcolor = "#FF00FF"
        graphline = "--"
        order = 666
        width = 2
    else:
        graphcolor = "#00FFFF"
        graphline = "-"
        order = 6666
        width = 3

    # Graph data
    times, Probs = timeyGraphdata(Hval, Hvec, State, Timey, res)
    times, TProbs = timeyGraphdata(HTval, HTvec, State, Timey, res)
    plt.plot(times, Probs, label="(F)", color=graphcolor, linestyle=graphline, zorder=order, linewidth=width)

    if epsilon == 0:
        plt.plot(times, TProbs, label="(T)", color="#FF0000", linestyle="-", zorder=0)


# plt.xlabel(r"time $(\varepsilon^{-1}")
# plt.ylabel(r"$\abs{c_{1}}^{2}")
# plt.title(r"$\ket{\Psi(0)}=\ket{\nu_{1}}$")
plt.xlabel("time (epsilon^(-1)")
plt.ylabel("Probability of staying in state v1")
plt.title("|phi(0)> = |v1>")
plt.legend()
plt.show()


# Clean up noise generated by approximation
# Threshold = 1e-10
# H.real[abs(H.real) < Threshold] = 0.0
# H.imag[abs(H.imag) < Threshold] = 0.0
# Hval.real[abs(Hval.real) < Threshold] = 0.0
# #Hval.imag[abs(Hval.imag) < Threshold] = 0.0
# Hvec.real[abs(Hvec.real) < Threshold] = 0.0
# Hvec.imag[abs(Hvec.imag) < Threshold] = 0.0
# H8x8.real[abs(H8x8.real) < Threshold] = 0.0
# H8x8.imag[abs(H8x8.imag) < Threshold] = 0.0
# H8x8val.real[abs(H8x8val.real) < Threshold] = 0.0
# H8x8val.imag[abs(H8x8val.imag) < Threshold] = 0.0
# H8x8vec.real[abs(H8x8vec.real) < Threshold] = 0.0
# H8x8vec.imag[abs(H8x8vec.imag) < Threshold] = 0.0

# Write Hamiltonian and it's diagonalizitation to .txt file.
# with open("NeutrinoToySolution.txt", "w") as txt_file:
    # txt_file.write(f"Assumed that big epsilon is equal to 1, for simplicity. Adjust for big epsilon equals (G_F / (sqrt(2)*V))\n\nSet values:\nTbar = {Tbar}\nomega = {omega}\ntheta = (1/2)arcsin(0.8)\nr = {r}\nphi = pi/4\nepsilon = {epsilon}\n")
    # # Full 8x8 solution
    # txt_file.write("\n\nFull Hamiltonian Solution (8x8 block): \n\nHval: ")
    # for i in range(len(H8x8val)):
        # txt_file.write(f"{H8x8val[i]} ")
    # txt_file.write("\n\nHvec: ")
    # for i in range(len(H8x8vec)):
        # txt_file.write(f"{H8x8vec[i]}\n")
    # txt_file.write("\n\n8x8 Hamiltonian: \n")
    # for i in range(len(H8x8)):
        # txt_file.write(f" {H8x8[i]}\n")
    # txt_file.write("\n\n")
    
    # # Full 28x28 solution
    # txt_file.write("\nFull Hamiltonian Solution: \n\nHval: ")
    # for i in range(len(Hval)):
        # txt_file.write(f"{Hval[i]} ")
    # txt_file.write("\n\nHvec: \n")
    # for i in range(len(Hvec)):
        # txt_file.write(f"{Hvec[i]}\n")
    # txt_file.write("\n\nFull Hamiltonian: \n")
    # for i in range(len(H)):
        # txt_file.write(f" {H[i]}\n")

