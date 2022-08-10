
#%%
import random 
import numpy as np
import math
import matplotlib.pyplot as plt
#Implement LJ energy (V) and forces (F) for a system of atoms in 3 dimensions.
sigma = 3.405
epsilon = 0.0103
pos_part = []
potentials_lj = []
force_lj=[]

nr_part = 2
D=3
i=0
j=1
N=1000
for i in (range(nr_part-1)):
    coords = np.random.rand(D)
    pos_part.append(coords)
    for particle in range((nr_part-1)):
        coords = np.random.rand(D)
        values= np.linspace(coords+abs(pos_part[i]+3),coords+abs(pos_part[i]+2)+4,N)
        for j in range(len(values)):
            r = abs(values[j]-pos_part[i])
            v= 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
            f = 4*epsilon*(12 *((sigma**12/(r**14)) - 6*( sigma**6 /(r**8))))*(values[j]-pos_part[i])
            force_lj.append(f)
            potentials_lj.append(v)
#print(values[0],pos_part)
#print(force_lj)

#print("LJ_force = ", f, "LJ_potential= ",potentials_lj)
plt.plot(values,force_lj)
plt.xlabel("separation of the particles")
plt.ylabel("Force [kg m/s^2]")
plt.figure()
plt.plot(values,potentials_lj)
plt.xlabel("separation of the particles")
plt.ylabel("LJ-potential [kg(m/s)^2]")
plt.figure()

#%%
#Velocity Verlet
def Velocity_Verlet(epsilon, sigma, m_1, m_2, N, h):
    kinetic_1 = []
    kinetic_2 = []
    kinetic = []

    position_1 =[np.random.rand(1)]
    position_2=[abs(np.random.rand(1))+4]

    print("intitial posiitons of the particles = ", position_1,position_2)
    
    v_1 = [0]
    v_2 = [0]
    
    potential = []
    hamiltonian= []
    tot_Energy = []


    for i in range(N):
        dist_1= abs(position_1[i]-position_2[i])
        
        force_1_1 =(4*epsilon*(12*(sigma**12/(dist_1**14)) - 6*( sigma**6 /(dist_1**8)))*(position_1[i]-position_2[i]))
        force_2_1 = (4*epsilon*(12*(sigma**12/(dist_1**14)) - 6*( sigma**6 /(dist_1**8)))*(position_2[i]-position_1[i]))
        
        a_1= force_1_1/m_1 
        v_half_1 = v_1[i] + (h*(a_1)/2)
        x_new = position_1[i] + h*v_half_1
        position_1.append(x_new)
        
        a_2 = force_2_1/m_2
        v_half_2 = v_2[i] + h*(a_2)/2
        x_new2 = position_2[i] + h*v_half_2
        a_2 = force_2_1/m_2
        position_2.append(x_new2)
        
        dist = abs(position_1[i+1]-position_2[i+1])

        force_1_2 =(4*epsilon*(12*(sigma**12/(dist**14)) - 6*( sigma**6 /(dist**8)))*(position_1[i+1]-position_2[i+1]))
        force_2_2 =(4*epsilon*(12*(sigma**12/(dist**14)) - 6*( sigma**6 /(dist**8)))*(position_2[i+1]-position_1[i+1]))
        
        a_step_1 = force_1_2/m_1
        a_step_2 = force_2_2/m_2
    
        v_1.append(v_1[i] + 0.5*(a_1 + a_step_1)*h)
        kinetic_1.append(m_1*(v_1[i]**2)/2)
        v_2.append(v_2[i]+ 0.5*(a_2 + a_step_2)*h)
        kinetic_2.append(m_2*(v_2[i]**2)/2)
        
        kinetic.append(kinetic_1[i]+kinetic_2[i])
        tot_Energy.append(4*epsilon*((sigma/dist_1)**12 - (sigma/dist_1)**6))
        hamiltonian.append(kinetic[i]+tot_Energy[i])

    return kinetic, potential,hamiltonian,position_1,position_2,tot_Energy, v_1,v_2
        


N = 1000
m_1= 1
m_2= 1
sigma = 3.405
epsilon = 0.0103
time=np.linspace(0,100,N)
delt = [0.01,0.1,1,3,4.5]

for h in delt:
    kinetic, potential,hamiltonian,position_1,position_2,tot_Energy,v_1,v_2 = Velocity_Verlet(epsilon, sigma,m_1,m_2,N,h)
    plt.plot(time,tot_Energy[0:N])
    plt.plot(time,kinetic[0:N])
    plt.plot(time,hamiltonian[0:N])
    plt.xlabel("time")
    plt.ylabel("Energy [kg(m/s)^2]")
    plt.legend(["Potential","Kinetic Energy","Hamiltonian"])
    plt.savefig("Energy_3")
    plt.figure
    plt.show()
    
    plt.plot(time,position_1[0:N])
    plt.plot(time,position_2[0:N])
    plt.xlabel("time")
    plt.ylabel("Distance r [m]")
    plt.legend(["Particle 1", "Particle_2"])
    plt.savefig("position_3")
    plt.figure
    plt.show()

    
    plt.plot(time,v_1[0:N])
    plt.plot(time,v_2[0:N])
    plt.xlabel("time")
    plt.ylabel("velocity [m/s]")
    plt.legend(["Particle 1", "Particle_2"])
    plt.savefig("velocity_3")
    plt.figure
    plt.show()

    

# %%
