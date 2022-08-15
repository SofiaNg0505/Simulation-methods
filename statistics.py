#%%
get_ipython().magic('reset -sf')
from tracemalloc import start
from numba import njit
from numba import types
from numba.extending import overload
import random

import random 
import numpy as np
from datetime import datetime
import math
import time 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
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

n = 8
N = n**3 
L = d*n
bins = 100
count = 0
sigma = 1
epsilon = 1
m = 1
M=100000
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
@njit
def acceleration(force,m):
    fastmath = True
    acc= force/m
    return acc
@njit
def v_half(i_vel, accel):
    fastmath = True
    v_half_step= i_vel + (accel)
    return v_half_step 
@njit
def new_pos(i_pos,  h, v_half_,L):
    fastmath = True
    x_new = i_pos + h*v_half_
    x_new = x_new%L
    return x_new 
@njit
def force(distance, absdistance):
    fastmath = True
    return (4*epsilon*(12*(sigma**12/(absdistance**14)) - 6*( sigma**6 /(absdistance**8)))*distance)

@njit
def MD(x,L,epsilon,sigma,dt,dr):
    fastmath = True
    h = np.zeros(bins) 
    acc_w = np.zeros((N,3))
    potential_particle = np.zeros((1))
    rs = np.arange(bins)*dr
    for k in range((N)-1):
        for j in range(k+1,(N)):  
            dist = x[k,:]- x[j,:]
            dist = dist- np.rint(dist/L)*L
            diff = np.sqrt(np.sum(dist**2))
            potential_particle = 4*epsilon*((sigma/diff)**12 - (sigma/diff)**6) + potential_particle
            forcee = force(dist,diff)
            idx = diff/dr
            h[int(idx)] = h[int(idx)] +1
            acc_w[k,:] = (acceleration(forcee, m)) + acc_w[k,:]
            acc_w[j,:]= (acceleration(-forcee, m)) + acc_w[j,:]

            
    h =h*  2/N
    rdf = (L**3 * h)/(N*4*np.pi*dr*rs**2) 
    
    return acc_w, potential_particle , rdf

#%%
start_time = datetime.now()
M = 100000
@njit
def ok(M):
    dt = 0.001
    v = np.random.uniform(-0.2,0.2,(N,3))
    x = pos
    potential_energy = []
    kinetic_energy = []
    hamiltonian = []
    radial_df = []
    temperature = []
    velocities = []
    positions = []

    for h in range(M): 
        print("timestep:", h )
        acc_w,potential_particle,rdf = MD(x , L, epsilon, sigma, dt,  L/bins)
        v_halff= v + dt*acc_w*0.5 
        x = (new_pos(x,dt,v_halff,L)) 
        positions.append(x)
        acc_h,potential,rdfnouse = MD(x , L, epsilon, sigma, dt, L/bins)
        sumofacc = sum(acc_w,acc_h)
        v = v+(dt*0.5*sumofacc)
        kinetic_particle= 0.5*m*(np.sum(v*v))

        tmp0 = 94.4
        tmp = tmp0/u_temp 
        T = 2*(kinetic_particle)/(N)/3*1/u_temp 
        #print("Temperature for timestep h" ,T*u_temp )
        if h<1000 and h%50== 0:
            v = v* np.sqrt(tmp/T)
            kinetic_particle = 0.5*m*(np.sum(v*v))
            T = 2*(kinetic_particle)/(N*3*1)

        potential_energy.append(potential_particle)
        kinetic_energy.append(kinetic_particle)
        radial_df.append(rdf)
        hamiltonian.append(kinetic_particle+potential_particle)
        temperature.append(T)
        velocities.append(v)
    return potential_energy,kinetic_energy, radial_df, hamiltonian, temperature,velocities 
potential_energy,kinetic_energy, radial_df, hamiltonian, temperature,velocities = ok(M)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
#%%
pott = []

for sublist in potential_energy:
    
    for val in sublist:

        pott.append(val)
print(pott)
#%%

pot = []
kin = []
ham= []
tem = []
vel =[]

for i in range(M):
    pot.append(np.round(pott[i]*e1,decimals = 2))
    kin.append(np.round(kinetic_energy[i]*e1,decimals = 2))
    tem.append(np.round(temperature[i],decimals = 2))
    ham.append(np.round(pott[i]*e1+kinetic_energy[i]*e1,decimals = 2))
    vel.append(np.round(np.sum(velocities[i]),decimals = 2))

np.savetxt("potential_energy_512_0", pot,fmt = '%f')
np.savetxt("kinetic_energy_512_0", kin,fmt = '%f')
np.savetxt("temperature_512_0",tem, fmt = '%f')
np.savetxt("velocities_512_0", vel,fmt = '%f')
np.savetxt("hamiiltonian_512_0", ham,fmt = '%f')

#%%

time = np.linspace(0,1000,M)

plt.plot(time,pot)
plt.plot(time,kin)
plt.plot(time,ham)
plt.legend(["kin","pot","hamiltonian"])
plt.ylabel("Energy [eV]]")
plt.xlabel("time")
plt.savefig("ENERGY512")
#%%
T = []
kinetic_energy = np.array(kinetic_energy)
#for i in kinetic_energy:
#    t = 2*(i)/(N*3*1)
#    T.append(t*u_temp)
#%%

dr = L/bins
rs = np.arange(bins)*dr
plt.plot(rs, np.sum(radial_df[400:],axis = 0)/(M-400))
plt.ylabel("g(r) [a.u.]")
plt.xlabel("r [Angstrom]")
plt.savefig("RDF9")
# %%
plt.plot(time,((T)))
plt.ylabel("Temperature [K]")
plt.xlabel("time ")
plt.savefig("temp")

#%%
T = temperature
eq = 400
kinetic_energy= np.sum(kinetic_energy[eq:])
sigma = np.sum( ((kinetic_energy/N)-((kinetic_energy/N)/(M-eq)))**2 )
cv = 1/((2/3*1) - (4*N * (sigma))/(9*1*((np.sum(T)/M)**2)))
print("Potential_energy", (np.sum(potential_energy[eq:])/(M-eq))*e1)
print("Kinetic_energy ", (np.sum(kinetic_energy)/(M-eq))*e1)
print("T ", (np.sum(T[eq:])/(M-eq)))
print("Cv", cv)
# %%
#Block analysis

potential_energy=np.loadtxt("Pot200k.txt")
kinetic_energy=np.loadtxt("Kin200k.txt")


starta = 69028
max = 200000
M=max-starta
block_size = []
n_b = []
#q = kinetic_energy[starta:max]*e1 #potential and kinetic energy
q= 2/3 * (kinetic_energy[starta:max]/125) * 119.8  #temperature

for times in range(16):
    block_size.append(2**times)
    n_b.append(round((max- starta)/block_size[times]))

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
