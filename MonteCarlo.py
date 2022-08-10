# MONTE CARLO
#%%
#Initialization
from cmath import exp
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit
import math 

s0 = 3.405e-10 # [m]
m0 = 39.948*1.6605402e-27 # [Kg]
k1 = 8.617333262e-5
k0 = 1.380649e-23 # [J/K]
e0 = 119.8*k0 # [J]
m1 = 0.03994 # [Kg/mol]
e1 = 119.8*k1
u_temp = e0/k0 # temperature units
u_len = s0 # length units

rho = 1.374e3 
d0 = (m0/rho)**(1/3) 
d = d0/u_len 

n = 10
N = n**3 
L = d*n
bins = 100
count = 0
sigma = 1
epsilon = 1
m = 1
pos = np.zeros((N,3))


for i1 in range(n):
    for i2 in range(n):
        for i3 in range(n):
            pos[count] = d*i1, d*i2, d*i3
            count +=1 
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, L, 1000)
xline = np.linspace(0, L, 1000)
yline = np.linspace(0, L, 1000)

# Data for three-dimensional scattered points
zdata = pos[:,0]
xdata = pos[:,1]
ydata = pos[:,2]
ax.scatter3D(xdata, ydata, zdata, ".", cmap='Greens')
plt.xlabel("Angstrom")
plt.ylabel("Angstrom")
plt.savefig("Lattice")
#%%
#functions

def random_displacement(pos,index):
    random_move= np.random.uniform(-0.3,0.3,3)
    #print(random_move)
    random_particle = pos[index]
    random_particle = random_particle + random_move
    return random_particle

def lj(x0, x1, epsilon, sigma,h):
    dist = x0-x1 
    dist = dist -np.rint(dist/L)*L
    diff = np.sqrt(np.sum(dist**2))
    potential = 4*epsilon*((sigma/diff)**12 - (sigma/diff)**6) 
    idx = diff/dr
    h[int(idx)] = h[int(idx)] +1
    return potential, h
#%% MC
print("start")
temp = (94.4)/119.8
#pos.remove(random_particle)

new_dis = 100
potential_energy = []
trial_move = 1
energy_squared = []
bins = 100
dr = L/bins
h = np.zeros((bins,(2)))
rs = np.arange(bins)*dr
rdff = []
for i in range(new_dis-1):
    index = (random.randrange(0,len(pos)))
    particle_old= pos[index] 
    random_particle= random_displacement(pos,index)
    pos = np.delete(pos, index,axis = 0)
    energy_old = np.zeros(1)
    energy_new = np.zeros(1)
    #print("positiion of particle old and particle new",particle_old,random_particle)
    
    for particles in range(N-1):
        print(pos[particles])
        energy_old, h[:,0] = lj(particle_old, pos[particles],epsilon,sigma,h[:,0]) + energy_old
        energy_new, h[:,1]  = lj(random_particle, pos[particles],epsilon,sigma,h[:,1]) + energy_new
        #delta_v = (energy_new*e1 -energy_old*e1) 
        delta_v = (energy_new - energy_old)
    #print("delta_v",delta_v)
    #print("enegy old and energy new",energy_old ,energy_new )
    if delta_v <= 0:
        pos=np.insert(pos,index,random_particle,axis = 0)
        trial_move =1 + trial_move
        print("accepted")
        potential_energy.append(energy_new*e1)
        energy_squared.append((e1*energy_new)**2)
        h[:,0] =h[:,0]*  2/N
        rdf = (L**3 * h[:,0])/(N*4*np.pi*rs**2) 
        rdff.append(rdf)
        #print("ok")
    else: 
        boltzmann_factor = math.exp(-delta_v/(temp))
        #print(boltzmann_factor,delta_v)
        #print("facc",boltzmann_factor)
        eff= np.random.uniform(0,1) 
        if eff<= boltzmann_factor:          
            #print(random_particle)                
            pos=np.insert(pos,index,random_particle,axis = 0)
            trial_move = 1+ trial_move
            potential_energy.append(energy_new*e1)
            energy_squared.append((e1*energy_new)**2)
            h[:,0] =h[:,0]*  2/N
            rdf = (L**3 * h[:,0])/(N*4*np.pi*rs**2) 
            rdff.append(rdf)
            #print("ok")
            print("accepted")
            
        else:
            #print(particle_old)
            pos=np.insert(pos,index,particle_old, axis= 0)
            print("denied")
            potential_energy.append(energy_old*e1)
            energy_squared.append((e1*energy_old)**2)
            h[:,1] =h[:,1]*  2/N
            h =h*  2/N
            rdf = (L**3 * h[:,0])/(N*4*np.pi*rs**2)
            rdff.append(rdf) 
        #print(pos)
       #print("accepted trial",trial_move)
        if i % 1000 == 0:
            print("Current step:",i)
            current = trial_move / (i+1)
            print("Current acceptance rate:",current)
#print(pos)
print("Acceptance rate:", trial_move/(new_dis))
print("end")
#%% compute values


# %%
plt.plot(potential_energy[0:1000],"--")
plt.xlabel(r'# trial')
plt.ylabel(r'Potential energy (eV)')
plt.show()
# %%
starta = 569
max = 2002
M=max-starta
block_size = []
n_b = []
q = potential_energy[starta:max] #potential 


for times in range(16):
    block_size.append(2**times)
    n_b.append(round((M)/block_size[times]))

average_quantitie = (1/M)*np.sum(q)
standard_error = np.zeros(len(block_size))
error_bar = np.zeros((2,len(block_size)))
block_step = np.linspace(1,16, 16)
aa = []
for k in range(len(block_size)):
    nr_box = round(len(q)/(block_size[k]))
    a = [q[x:x+block_size[k]] for x in range(0, len(q), block_size[k])]
    a_i = []
    summa = 0
    for block in range(len(a)):
        a_ii = ((np.sum(a[block]))) #summation of each block
        a_i.append((1/(block_size[k])*a_ii))
        summa = ((a_i[block]- average_quantitie)**2) + summa
    sigma_ak = np.sqrt((1/((n_b[k])))*summa)
    standard_error[k] = sigma_ak/(np.sqrt(n_b[k])-1)
    error_bar[0,k]= (1/(np.sqrt(2*(n_b[k])-1)))
    error_bar[1,k]= (1/(np.sqrt(2*(n_b[k])-1)))
plt.errorbar(block_step,standard_error, yerr = standard_error*error_bar, fmt='.k',color ="g")
plt.xlabel("number of block transformation applied")
plt.ylabel("sigma(t_average) eV")
plt.savefig("sigmat")
# %%
dr = L/bins
rs = np.arange(bins)*dr
plt.plot(rs, np.sum(rdff[starta:],axis = 0)/(M))
plt.ylabel("g(r) [a.u.]")
plt.xlabel("r [Angstrom]")
plt.savefig("RDF9")

# %%
