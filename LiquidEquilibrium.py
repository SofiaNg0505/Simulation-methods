#%%
from IPython import get_ipython
get_ipython().magic('reset -sf')
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

rho = 1.4e3 
d0 = (m0/rho)**(1/3) 
d = d0/u_len 

n = 3
N = n**3 
L = d*n
bins = 100
count = 0
sigma = 1
epsilon = 1
m = 1
pos = np.zeros((N,3))
count = 0
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

def acceleration(force,m):
     acc= force/m
     return acc
 
def v_half(i_vel, accel):
    v_half_step= i_vel + (accel)
    return v_half_step 

def new_pos(i_pos,  h, v_half_,L):
    x_new = i_pos + h*v_half_
    x_new = x_new%L
    return x_new 

def force(distance, absdistance):
    return (4*epsilon*(12*(sigma**12/(absdistance**14)) - 6*( sigma**6 /(absdistance**8)))*distance)
#x = np.zeros((5000,N,3)) 

def MD(x,L,epsilon,sigma,dt,dr):
    h = np.zeros(bins) 
    acc_w = np.zeros((N,3))
    potential_particle = np.zeros((1))
    rs = np.arange(bins)*dr
    x_new= np.zeros((N,3))
    for k in range((N)-1):
        for j in range(k+1,(N)):   
            #print("j=",j)
            dist = x[k,:]- x[j,:]
            dist = dist- np.rint(dist/L)*L
            diff = np.linalg.norm(dist)
            potential_particle = 4*epsilon*((sigma/diff)**12 - (sigma/diff)**6) + potential_particle
            forcee = force(dist,diff)
            idx = diff/dr
            h[int(idx)] = h[int(idx)] +1
            acc_w[k,:] = (acceleration(forcee, m)) + acc_w[k,:]
            acc_w[j,:]= (acceleration(-forcee, m)) + acc_w[j,:]
            
    h *=  2/N
    rdf = (L**3 * h)/(N*4*np.pi*dr*rs**2) 
    return acc_w, potential_particle , rdf
#%%
start_time = datetime.now()

sumv = np.zeros((N,3))
v = np.random.uniform(-0.2,0.2,(N,3))
x = pos
potential_energy = []
M = 2000
kinetic_energy = []
hamiltonian = []
radial_df = []
temperature = []
velocities = []
positions = []
for h in range(M): 
    print("timestep: ",h)
    dt = 0.001
    acc_w,potential_particle,rdf = MD(x , L, epsilon, sigma, dt,  L/bins)
    v_halff= v + dt*acc_w*0.5 
    x = (new_pos(x,dt,v_halff,L)) 
    positions.append(x)
    acc_h,potential,rdfnouse = MD(x , L, epsilon, sigma, dt, dr = L/bins)

    sumofacc = sum(acc_w,acc_h)
    v = v+(dt*0.5*sumofacc)
    kinetic_particle= 0.5*m*(np.sum(v*v))

    tmp0 = 95 
    tmp = tmp0/u_temp 
    T = 2*(kinetic_particle)/(N)/3*1/u_temp 
    #print("Temperature for timestep h" ,T*u_temp )
    if h<1500 and h%50== 0:
        v = v* np.sqrt(tmp/T)
        kinetic_particle = 0.5*m*(np.sum(v*v))
        T = 2*(kinetic_particle)/(N*3*1)

        #print( T) # Reduced)
    #print("kinetic_energy", kinetic_particle)
    #print("potential_energy",potential_energy)
    potential_energy.append(potential_particle)
    kinetic_energy.append(kinetic_particle)
    radial_df.append(rdf)
    hamiltonian.append(kinetic_particle+potential_particle)
    temperature.append(T)
    velocities.append(v)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

#%%

time = np.linspace(0,1000,2000)
plt.plot(time,kinetic_energy[:]*e1)
plt.plot(time,potential_energy[:]*e1)
plt.plot(time,hamiltonian[:]*e1)
plt.legend(["kin","pot","hamiltonian"])
plt.ylabel("Energy [eV]]")
plt.xlabel("time")
plt.savefig("ENERGY512")
#%%
T = []
kinetic_energy = np.array(kinetic_energy[:])
for i in kinetic_energy:
    t = 2*(i)/(N*3*1)
    T.append(t*u_temp)
#%%

dr = L/bins
rs = np.arange(bins)*dr
plt.plot(rs, np.sum(g,axis = 0)/M)
plt.ylabel("g(r) [a.u.]")
plt.xlabel("r [Angstrom]")
plt.savefig("RDF9")


# %%
plt.plot(time,((T)))
plt.ylabel("Temperature [K]")
plt.xlabel("time ")
plt.savefig("temp")

#%%
M=2000
eq = 400
kinetic_energy= np.sum(kinetic_energy[eq:])
sigma = np.sum( ((kinetic_energy/N)-((kinetic_energy/N)/(M-eq)))**2 )
cv = 1/((2/3*1) - (4*N * (sigma))/(9*1*((np.sum(T)/M)**2)))
print("Potential_energy", (np.sum(potential_energy[eq:])/(M-eq))*e1)
print("Kinetic_energy ", (np.sum(kinetic_energy)/(M-eq))*e1)
print("T ", (np.sum(T[eq:])/(M-eq)))
print("Cv", cv)
# %%
