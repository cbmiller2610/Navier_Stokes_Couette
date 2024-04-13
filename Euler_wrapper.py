import numpy as np
import pdb
#from ctypes import *
import timeit

#Functions
def grid_transform(xi, eta, m1, m2, nozzle_xloc, nozzle_ymax, throat_xloc, throat_ymax):

    #Region 1: converging
    if xi <= 5:
        y = eta * (m1 * (xi - inlet_xloc) + inlet_ymax)
    #Region 2: diverging
    elif (xi > 5):
        y = eta*(m2 * (xi - throat_xloc) + throat_ymax)
    else:
        print("OUT OF RANGE")
    return y

def initial_cond(xi):
    #s1 = 0.5 * (10 / 3)
    #s2 = 1.5 * (10 / 3)
    #s3 = 2.1 * (10 / 3)
    #s4 = 3 * (10 / 3)
    #if (xi >= 0) & (xi < s1):
    #    rho = 1.0
    #    T = 1.0
    #elif (xi >= s1) & (xi < s2):
    #    rho = 1.0 - 0.366 * (3 / 10) * (xi - s1)
    #    T = 1.0 - 0.167 * (3 / 10) * (xi - s1)
    #elif (xi >= s2) & (xi < s3):
    #    rho = 0.634 - 0.702 * (3 / 10) * (xi - s2)
    #    T = 0.833 - 0.4908 * (3 / 10) * (xi - s2)
    #elif (xi >= s3) & (xi <= s4):
    #    rho = 0.5892 + 0.10228 * (3 / 10) * (xi - s3)
    #    T = 0.93968 + 0.0622 * (3 / 10) * (xi - s3) 
    #else:
    #    print("OUT OF RANGE")
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
    U_step[:, :, 3] = (g - 1) * (U4 - (g / 2) * ((U2**2 + U3**2) / U1)) #pressure
    U_step[:, :, 4] = (g - 1) * (U4 / U1 - (g / 2) * ((U2**2 + U3**2) / U1**2)) #temperature
    return U_step

def U_to_E(U, xi_div, eta_div):
    U1 = U[:, :, 0]
    U2 = U[:, :, 1]
    U3 = U[:, :, 2]
    U4 = U[:, :, 3]
    E_step = np.empty((xi_div, eta_div, 4))
    g = 1.4
    E_step[:, :, 0] = U2
    E_step[:, :, 1] = U2**2 / U1 + (1 - 1 / g) * (U4 - (g / 2) * ((U2**2 + U3**2) / U1))
    E_step[:, :, 2] = (U2 * U3) / U1
    E_step[:, :, 3] = (g * U2 * U4) / U1 - ((g * (g - 1)) / 2) * ((U2**3 + U2 * U3**2)/ (U1**2))
    return E_step

def U_to_F(U, xi_div, eta_div):
    U1 = U[:, :, 0]
    U2 = U[:, :, 1]
    U3 = U[:, :, 2]
    U4 = U[:, :, 3]
    F_step = np.empty((xi_div, eta_div, 4))
    g = 1.4
    F_step[:, :, 0] = U3
    F_step[:, :, 1] = (U2 *U3) / U1
    F_step[:, :, 2] = U3**2 / U1 + (1 - 1 / g) * (U4 - (g / 2) * ((U2**2 + U3**2) / U1))
    F_step[:, :, 3] = (g * U3 * U4) / U1 - ((g * (g - 1)) / 2) * ((U2**2 * U3 + U3**3) / (U1**2))
    return F_step

def Ughost_to_Eghost(U, xi_div):
    U1 = U[:, :, 0]
    U2 = U[:, :, 1]
    U3 = U[:, :, 2]
    U4 = U[:, :, 3]
    E_step = np.empty((xi_div-2, 2, 4))
    g = 1.4
    E_step[:, :, 0] = U2
    E_step[:, :, 1] = U2**2 / U1 + (1 - 1 / g) * (U4 - (g / 2) * ((U2**2 + U3**2) / U1))
    E_step[:, :, 2] = (U2 * U3) / U1
    E_step[:, :, 3] = (g * U2 * U4) / U1 - ((g * (g - 1)) / 2) * ((U2**3 + U2 * U3**2) / (U1**2))
    return E_step

def Ughost_to_Fghost(U, xi_div):
    U1 = U[:, :, 0]
    U2 = U[:, :, 1]
    U3 = U[:, :, 2]
    U4 = U[:, :, 3]
    F_step = np.empty((xi_div-2, 2, 4))
    g = 1.4
    F_step[:, :, 0] = U3
    F_step[:, :, 1] = (U2 * U3) / U1
    F_step[:, :, 2] = U3**2 / U1 + (1 - 1 / g) * (U4 - (g / 2) * ((U2**2 + U3**2) / U1))
    F_step[:, :, 3] = (g * U3 * U4) / U1 - ((g * (g - 1)) / 2) * ((U2**2 * U3 + U3**3) / (U1**2))
    return F_step

def grid_transform(xi, eta, m1, m2, nozzle_xloc,nozzle_ymax,throat_xloc,throat_ymax):


    #Region 1: converging
    if xi<=5:
        y=eta * (m1 * (xi - inlet_xloc) + inlet_ymax)
    #Region 2: diverging
    elif (xi > 5):
        y=eta * (m2 * (xi-throat_xloc) + throat_ymax)
    #Region 3: farfield
    # elif (xi > 5) & (xi <= 25):
    #     y=eta*(m3*(xi-nozzle_xloc)+nozzle_ymax)
    else:
        print("WRONG VALUE")
    return y

def coord_transform(U_np_3D, E_np_3D, F_np_3D, xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax):
    if xi_grid <= 5:
        ys = (m1 * (xi_grid - inlet_xloc) + inlet_ymax)
        m=m1
    elif xi_grid > 5:
        ys = (m2 * (xi_grid - throat_xloc) + throat_ymax)
        m=m2
    else:
        print("WRONG VALUE")
    U_new = U_np_3D * ys
    E_new = E_np_3D * ys
    F_new = -m * eta_grid * E_np_3D + F_np_3D
    return U_new, E_new, F_new

def return_transform(U_np_3D, E_np_3D, F_np_3D, xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax):
    if xi_grid <= 5:
        ys = (m1 * (xi_grid - inlet_xloc) + inlet_ymax)
        m=m1
    elif xi_grid > 5:
        ys = (m2 * (xi_grid - throat_xloc) + throat_ymax)
        m=m2
    else:
        print("WRONG VALUE")
    U_new = U_np_3D / ys
    E_new = E_np_3D / ys
    F_new = F_np_3D + (m * eta_grid * E_np_3D) / ys
    return U_new, E_new, F_new



#Import C dlls
#solver_dll = CDLL("./euler_solver.so")
#solver_dll.argtypes = [c_double, c_double, c_double, c_int, c_int, c_int]

#Gridding Inputs
xi_max = 10
eta_max = 1
xi_div = 501
eta_div = 251
dxi = xi_max / (xi_div - 1)
deta = eta_max / (eta_div - 1)
dt = 0.0001
tsteps = 2000 

#Grid Generation
xi_vec = np.linspace(0,xi_max,xi_div)
eta_vec = np.linspace(0,eta_max,eta_div)

xi_grid,eta_grid = np.meshgrid(xi_vec,eta_vec,indexing='ij')
x_grid = xi_grid

grid_trans_func = np.vectorize(grid_transform)
throat_ymax = 1
inlet_ymax = np.sqrt(5.95)
nozzle_ymax = np.sqrt(5.95)

throat_xloc = 5
inlet_xloc = 0
nozzle_xloc = 10


m1 = (throat_ymax - inlet_ymax)/(throat_xloc - inlet_xloc)
m2 = (nozzle_ymax - throat_ymax)/(nozzle_xloc - throat_xloc)

y_grid = grid_trans_func(xi_grid,eta_grid, m1, m2, nozzle_xloc,nozzle_ymax,throat_xloc,throat_ymax)

#Initial Conditions

IC_func = np.vectorize(initial_cond)
rho, T = IC_func(x_grid)
#u = (0.59 / rho)*np.cos(np.arctan2(y_grid, 10))
#v = (0.59 / rho)*np.sin(np.arctan2(y_grid, 10))
u = (0.01 / rho)*np.cos(np.arctan2(y_grid, 10))
v = (0.01 / rho)*np.sin(np.arctan2(y_grid, 10))

v[0:int((xi_div-1)/2+1),:] = -1*v[0:int((xi_div-1)/2+1),:]
g = 1.4

#Creation of the Tensors of Conserved Variables and Fluxes as C Arrays
#tensize = xi_div * eta_div * 4
#tentype = c_double * tensize
#U = tentype()
#U_ptr = byref(U)
#E = tentype()
#E_ptr = byref(E)
#F = tentype()
#F_ptr = byref(F)
#U_np_1D = np.ctypeslib.as_array(U)
#U_np_3D = np.reshape(U_np_1D, (xi_div, eta_div, 4))
#E_np_1D = np.cytypeslib.as_array(E)
#E_np_3D = np.reshape(E_np_1D, (xi_div, eta_div, 4))
#F_np_1D = np.cytypeslib.as_array(F)
#F_np_3D = np.reshape(F_np_1D, (xi_div, eta_div, 4))
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

P_stor[:, :, :, 0] = U_to_prim(U_np_3D, xi_div, eta_div)
#Transform from Cartesian to Computational Grid
coord_transform_func = np.vectorize(coord_transform)
return_transform_func = np.vectorize(return_transform)
U_np_3D[:, :, 0], E_np_3D[:, :, 0], F_np_3D[:, :, 0] = coord_transform_func(U_np_3D[:, :, 0], E_np_3D[:, :, 0], F_np_3D[:, :, 0], xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax)
U_np_3D[:, :, 1], E_np_3D[:, :, 1], F_np_3D[:, :, 1] = coord_transform_func(U_np_3D[:, :, 1], E_np_3D[:, :, 1], F_np_3D[:, :, 1], xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax)
U_np_3D[:, :, 2], E_np_3D[:, :, 2], F_np_3D[:, :, 2] = coord_transform_func(U_np_3D[:, :, 2], E_np_3D[:, :, 2], F_np_3D[:, :, 2], xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax)
U_np_3D[:, :, 3], E_np_3D[:, :, 3], F_np_3D[:, :, 3] = coord_transform_func(U_np_3D[:, :, 3], E_np_3D[:, :, 3], F_np_3D[:, :, 3], xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax)

U_inlet = U_to_prim(U_np_3D, xi_div, eta_div)
T_inlet = U_inlet[0, 1, 4]
rho_inlet = U_inlet[0,1,0]

with open('TEST_U_after.dat', 'wb') as f:
    np.save(f, U_np_3D)

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
    #Eghosts = Ughost_to_Eghost(Ughosts, xi_div)
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
    
    #Transform Back to Physical Coordinates

    U_stor[:, :, 0], E_stor[:, :, 0], F_stor[:, :, 0] = return_transform_func(U_np_3D[:, :, 0], E_np_3D[:, :, 0], F_np_3D[:, :, 0], xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax)
    U_stor[:, :, 1], E_stor[:, :, 1], F_stor[:, :, 1] = return_transform_func(U_np_3D[:, :, 1], E_np_3D[:, :, 1], F_np_3D[:, :, 1], xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax)
    U_stor[:, :, 2], E_stor[:, :, 2], F_stor[:, :, 2] = return_transform_func(U_np_3D[:, :, 2], E_np_3D[:, :, 2], F_np_3D[:, :, 2], xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax)
    U_stor[:, :, 3], E_stor[:, :, 3], F_stor[:, :, 3] = return_transform_func(U_np_3D[:, :, 3], E_np_3D[:, :, 3], F_np_3D[:, :, 3], xi_grid, eta_grid, m1, m2, inlet_xloc, inlet_ymax, throat_xloc, throat_ymax)


    P_stor[:, :, :, i] = U_to_prim(U_stor, xi_div, eta_div)

with open('TEST.dat', 'wb') as f: 
    np.save(f, P_stor)
#with open('ustor.dat','wb') as f:
#    np.save(f, U_stor)
