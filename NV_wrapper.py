import numpy as np
import pdb
#from ctypes import *
import timeit

def initial_cond(xi):
    rho = 1 
    T = 1
    return rho, T

def U_to_prim(U, xi_div, eta_div):
    #returns rho, u, v, and p primitive variable arrays
    U1 = U[:, :, 0]
    U2 = U[:, :, 1]
    U3 = U[:, :, 2]
    U4 = U[:, :, 3]
    U_step = np.empty((xi_div, eta_div, 5))
    g = 1.4
    U_step[:, :, 0] = U1 #rho
    U_step[:, :, 1] = U2/U1 #u
    U_step[:, :, 2] = U3/U1 #v
    U_step[:, :, 3] = (g - 1) * (U4 - (U2**2 + U3**2) / (2*U1))) #pressure
    U_step[:, :, 4] = U4 / U1 - ((U2**2 + U3**2) / (2*U1**2)) #temperature
    return U_step
#Gridding Inputs
xi_max = 10
eta_max = 1
xi_div = 501
eta_div = 51
dxi = xi_max / (xi_div - 1)
deta = eta_max / (eta_div - 1)
dt = 0.0
tsteps = 2000 

#Grid Generation
xi_vec = np.linspace(0,xi_max,xi_div)
eta_vec = np.linspace(0,eta_max,eta_div)

xi_grid,eta_grid = np.meshgrid(xi_vec,eta_vec,indexing='ij')

#Initial Conditions

rho = np.ones(xi_div,eta_div)
u = np.zeros(xi_div,eta_div)
u[
e = np.full((xi_div,eta_div), (0.1712*519)/(3350^2))

g = 1.4

P_stor = np.empty((xi_div, eta_div, 5, tsteps+1)) #store rho, u, v, p, and T values
U_stor = np.empty((xi_div, eta_div, 4))
E_stor = np.empty((xi_div, eta_div, 4))
F_stor = np.empty((xi_div, eta_div, 4))
U_np_3D = np.empty((xi_div, eta_div, 4))
Upred = np.empty((xi_div, eta_div, 4))
Ucorr = np.empty((xi_div, eta_div, 4))
E_np_3D = np.empty((xi_div, eta_div, 4))
F_np_3D = np.empty((xi_div, eta_div, 4))

U1 = rho
U2 = rho * u
U3 = rho * v
U4 = rho * (T / (g - 1) + (g / 2) * (u**2 + v**2))

U_np_3D[:, :, 0] = U1
U_np_3D[:, :, 1] = U2
U_np_3D[:, :, 2] = U3
U_np_3D[:, :, 3] = U4

E_np_3D[:, :, 0] = U2
E_np_3D[:, :, 1] = U2**2 / U1 + (1 - 1 / g) * (U4 - (g / 2) * ((U2**2 + U3**2) / U1))
E_np_3D[:, :, 2] = (U2 * U3) / U1
E_np_3D[:, :, 3] = (g * U2 * U4) / U1 - ((g * (g - 1)) / 2) * ((U2**3 + U2 * U3**2) / (U1**2))

F_np_3D[:, :, 0] = U3
F_np_3D[:, :, 1] = (U2 * U3) / U1
F_np_3D[:, :, 2] = U3**2 / U1 + (1 - 1 / g) * (U4 - (g / 2) * ((U2**2 + U3**2) / U1))
F_np_3D[:, :, 3] = (g * U3 * U4) / U1 - ((g * (g - 1)) / 2) * ((U2**2 * U3 + U3**3) / (U1**2))

with open('TEST_U_before.dat', 'wb') as f:
    np.save(f, U_np_3D)

P_stor[:, :, :, 0] = U_to_prim(U_np_3D, xi_div, eta_div)i

Ughosts = np.empty((xi_div-2, 2, 4))
Eghosts = np.empty((xi_div-2, 2, 4))
Fghosts = np.empty((xi_div-2, 2, 4))

for i in range(1,tsteps):
    #Update boundary nodes from last timestep
    #left boundary, subsonic inlet (2 float, 2 prescribed)
    U_np_3D[0, :, 0] = rho_inlet
    U_np_3D[0, :, 1:3] = 2 * U_np_3D[1, :, 1:3] - U_np_3D[2, :, 1:3]
    u0 = U_np_3D[0, :, 1] / U_np_3D[0, :, 0]
    v0 = U_np_3D[0, :, 2] / U_np_3D[0, :, 0]
    U_np_3D[0, :, 3] = rho_inlet * (T_inlet / (g - 1) + (g / 2) * (u0**2 + v0**2))

    #right boundary, supersonic outlet, all quantities float
    U_np_3D[-1, :, :] = 2 * U_np_3D[-2, :, :] - U_np_3D[-3, :, :]
    #Wall and symmetry boundaries (update ghosts)
    #Lower Boundary
    Ughosts[:, 0, :] = U_np_3D[1:-1, 1, :]
    Ughosts[:, 0, 2] = - U_np_3D[1:-1, 1, 2]
    #Upper Boundary
    Ughosts[:, 1, :] = U_np_3D[1:-1, -2, :]
    Ughosts[:, 1, 2] = - U_np_3D[1:-1, -2, 2]

    #Update Flux Terms
    E_np_3D = U_to_E(U_np_3D, xi_div, eta_div)
    F_np_3D = U_to_F(U_np_3D, xi_div, eta_div)
    Fghosts = Ughost_to_Fghost(Ughosts, xi_div)

    #Predictor Step (Upper boundary uses ghost)
    #Update Upper Before Internal Flow
    Upred = U_np_3D
    Upred[1:-1, -1, :]  = (U_np_3D[1:-1, -1, :]
                            - (dt / dxi) * (E_np_3D[2:xi_div, -1, :] - E_np_3D[1:-1, -1, :])
                            - (dt / deta) * (Fghosts[:, 1, :] - F_np_3D[1:-1, -1, :]))
    #Inner
    Upred[1:-1, 0:-1, :] = (U_np_3D[1:-1, 0:-1, :]
                            - (dt / dxi) * (E_np_3D[2:xi_div, 0:-1, :] - E_np_3D[1:-1, 0:-1, :])
                            - (dt / deta) * (F_np_3D[1:-1, 1:eta_div, :] - F_np_3D[1:-1, 0:-1, :]))

    #Corrector Step (Lower boundary uses ghost)
    #Update Flux Terms from Predictor
    E_np_3D = U_to_E(Upred, xi_div, eta_div)
    F_np_3D = U_to_F(Upred, xi_div, eta_div)
    #Update Lower Before Internal Flow
    Ucorr = U_np_3D
    Ucorr[1:-1, 0, :]  = (U_np_3D[1:-1, 0, :]
                            - (dt / dxi) * (E_np_3D[1:-1, 0, :] - E_np_3D[0:-2, 0, :])
                            - (dt / deta) * (F_np_3D[1:-1, 0, :] - Fghosts[:, 0, :]))
    #Inner
    Ucorr[1:-1, 1:eta_div, :] = (U_np_3D[1:-1, 1:eta_div, :]
                                 - (dt / dxi) * (E_np_3D[1:-1, 1:eta_div, :] - E_np_3D[0:-2, 1:eta_div, :])
                                 - (dt / deta) * (F_np_3D[1:-1, 1:eta_div, :] - F_np_3D[1:-1, 0:-1, :]))

    #Updator
    U_np_3D = 0.5 * (Upred + Ucorr)

    #Update Flux Terms
    E_np_3D = U_to_E(U_np_3D, xi_div, eta_div)
    F_np_3D = U_to_F(U_np_3D, xi_div, eta_div)
    

with open('TEST.dat', 'wb') as f: 
    np.save(f, P_stor)
#with open('ustor.dat','wb') as f:
#    np.save(f, U_stor)
