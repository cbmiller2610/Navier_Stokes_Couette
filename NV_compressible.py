import numpy as np
import timeit
import pdb
import csv

def read_config(filename):
    with open(filename) as f:
        dict_reader = csv.DictReader(f, delimiter=',')
        read_list = list(dict_reader)
    return read_list

def init_grid(config):
    x_max = float(config['x_max'])
    y_max = float(config['y_max'])
    x_div = int(config['x_div'])
    y_div = int(config['y_div'])
    Reh = float(config['Reh'])
    g = float(config['g'])
    Pr = float(config['Pr'])
    M = float(config['M'])

    dx = x_max/(x_div-1)
    dy = y_max/(y_div-1)

    x_vec = np.linspace(0,x_max,x_div)
    y_vec = np.linspace(0,y_max,y_div)

    x_grid,y_grid = np.meshgrid(x_vec,y_vec,indexing='ij')

    grid = {'x_max':x_max,
            'y_max':y_max, 
            'x_div':x_div,
            'y_div':y_div,
            'dx':dx,
            'dy':dy,
            'x_grid':x_grid,
            'y_grid':y_grid
            }

    #Initial Conditions
    rho = np.ones((x_div,y_div))
    u = np.full((x_div,y_div),0.001)
    u[:,-1] = np.ones(x_div)
    v = np.full((x_div,y_div),0.001)
    T = np.ones((x_div,y_div))
    e = T/(g*(M**2)*(g-1))
    p = (g-1)*rho*e
    mu = np.ones((x_div,y_div))

    flow = {'rho':rho,
            'u':u,
            'v':v,
            'T':T,
            'e':e,
            'p':p,
            'mu':mu,
            'Reh':Reh,
            'g':g,
            'Pr':Pr,
            'M':M
            }
    return grid,flow

def calc_tstep(grid,flow):
    sigma = 0.6
    a = np.sqrt((flow['g']*flow['p'])/flow['rho'])
    dt_CFL = (
            (abs(flow['u'])/grid['dx']) + (abs(flow['v'])/grid['dy'])+
            a*np.sqrt((1/(grid['dx']**2)) + (1/(grid['dy']**2)))
            )
    Rex = (flow['rho']*abs(flow['u'])*grid['dx'])/flow['mu']
    Rey = (flow['rho']*abs(flow['v'])*grid['dy'])/flow['mu']

    Re_min = np.minimum(Rex[1:-1,1:-1],Rey[1:-1,1:-1])

    dt = np.min((sigma*dt_CFL[1:-1,1:-1])/(1+2/Re_min))
    #pdb.set_trace()
    dt = 0.0001
    #dt = 0.01
    return dt

def mac_predictor(grid,flow,dt):
    U_np_3D = encode_U(flow,grid)
    qE = heat_trans_pred_E(grid,flow)
    qF = heat_trans_pred_F(grid,flow)
    tauxx,tauxy_E = viscous_pred_E(grid,flow)
    tauxy_F,tauyy = viscous_pred_F(grid,flow)
    E_np_3D = encode_E(flow,qE,tauxx,tauxy_E,grid)
    F_np_3D = encode_F(flow,qF,tauxy_F,tauyy,grid)

    U_np_3D[1:-1,1:-1,:] = (U_np_3D[1:-1, 1:-1, :]
                            - (dt / grid['dx']) * (E_np_3D[2:, 1:-1, :] - E_np_3D[1:-1, 1:-1, :])
                            - (dt / grid['dy']) * (F_np_3D[1:-1, 2:, :] - F_np_3D[1:-1, 1:-1, :]))
    
    flow_out = decode_U(U_np_3D,flow)
    return flow_out

def mac_corrector(grid,flow,flow_pred,dt):
    U_corr = np.empty((grid['x_div'], grid['y_div'],4))
    U_pred = encode_U(flow_pred,grid)
    U_init = encode_U(flow,grid)
    qE = heat_trans_corr_E(grid,flow_pred)
    qF = heat_trans_corr_F(grid,flow_pred)
    tauxx,tauxy_E = viscous_corr_E(grid,flow_pred)
    tauxy_F,tauyy = viscous_corr_F(grid,flow_pred)
    E_np_3D = encode_E(flow_pred,qE,tauxx,tauxy_E,grid)
    F_np_3D = encode_F(flow_pred,qF,tauxy_F,tauyy,grid)

    U_corr[1:-1,1:-1,:] = 0.5*(U_init[1:-1,1:-1,:] + U_pred[1:-1, 1:-1, :]
                                 - (dt / grid['dx']) * (E_np_3D[1:-1, 1:-1, :] - E_np_3D[0:-2, 1:-1, :])
                                 - (dt / grid['dy']) * (F_np_3D[1:-1, 1:-1, :] - F_np_3D[1:-1, 0:-2, :]))
    flow_out = decode_U(U_corr,flow)
    return flow_out

def mac_combine(grid,flow_pred,flow_corr):
    U_pred = encode_U(flow_pred,grid)
    U_corr = encode_U(flow_corr,grid)

    U_final = 0.5*(U_pred+U_corr)

    flow_out = decode_U(U_final,flow_corr)
    return flow_out

def viscous_pred_E(grid,flow):
    dudx = np.empty((grid['x_div'], grid['y_div']))
    dudy = np.empty((grid['x_div'], grid['y_div']))
    dvdx = np.empty((grid['x_div'], grid['y_div']))
    dvdy = np.empty((grid['x_div'], grid['y_div']))
    u = flow['u']
    v = flow['v']
    mu = flow['mu']
    Reh = flow['Reh']
    dxi = grid['dx']
    deta = grid['dy']

    dudx[1:,:] = ((u[1:,:]-u[0:-1,:])/dxi)#rear diff
    dudx[0,:] = ((u[1,:]-u[0,:])/dxi)# fwd diff at left boundary only

    dvdx[1:,:] = ((v[1:,:]-v[0:-1,:])/dxi)#rear diff
    dvdx[0,:] = ((v[1,:]-v[0,:])/dxi)# fwd diff at left boundary only

    dudy[:,1:-1] = ((u[:,2:]-u[:,0:-2])/(2*deta)) #central diff
    dudy[:,0] = ((u[:,1]-u[:,0])/(deta)) #fwd diff at boundary only
    dudy[:,-1] = ((u[:,-1]-u[:,-2])/(deta)) #rear diff at boundary only

    dvdy[:,1:-1] = ((v[:,2:]-v[:,0:-2])/(2*deta)) #central diff
    dvdy[:,0] = ((v[:,1]-v[:,0])/(deta)) #fwd diff at boundary only
    dvdy[:,-1] = ((v[:,-1]-v[:,-2])/(deta)) #rear diff at boundary only

    tauxx = ((2*mu)/(3*Reh))*(2*dudx-dvdy)
    tauxy =  (mu/Reh)*(dudy+dvdx)
    
    return tauxx, tauxy

def viscous_corr_E(grid,flow):
    dudx = np.empty((grid['x_div'], grid['y_div']))
    dudy = np.empty((grid['x_div'], grid['y_div']))
    dvdx = np.empty((grid['x_div'], grid['y_div']))
    dvdy = np.empty((grid['x_div'], grid['y_div']))
    u = flow['u']
    v = flow['v']
    mu = flow['mu']
    Reh = flow['Reh']
    dxi = grid['dx']
    deta = grid['dy']

    dudx[0:-1,:] = ((u[1:,:]-u[0:-1,:])/dxi)#fwd diff
    dudx[-1,:] = ((u[-1,:]-u[-2,:])/dxi)# rear diff at right boundary only

    dvdx[0:-1,:] = ((v[1:,:]-v[0:-1,:])/dxi)#fwd diff
    dvdx[-1,:] = ((v[-1,:]-v[-2,:])/dxi)# rear diff at right boundary only

    dudy[:,1:-1] = ((u[:,2:]-u[:,0:-2])/(2*deta)) #central diff
    dudy[:,0] = ((u[:,1]-u[:,0])/(deta)) #fwd diff at boundary only
    dudy[:,-1] = ((u[:,-1]-u[:,-2])/(deta)) #rear diff at boundary only

    dvdy[:,1:-1] = ((v[:,2:]-v[:,0:-2])/(2*deta)) #central diff
    dvdy[:,0] = ((v[:,1]-v[:,0])/(deta)) #fwd diff at boundary only
    dvdy[:,-1] = ((v[:,-1]-v[:,-2])/(deta)) #rear diff at boundary only

    tauxx = ((2*mu)/(3*Reh))*(2*dudx-dvdy)
    tauxy =  (mu/Reh)*(dudy+dvdx)

    return tauxx, tauxy

def viscous_pred_F(grid,flow):
    dudx = np.empty((grid['x_div'], grid['y_div']))
    dudy = np.empty((grid['x_div'], grid['y_div']))
    dvdx = np.empty((grid['x_div'], grid['y_div']))
    dvdy = np.empty((grid['x_div'], grid['y_div']))
    u = flow['u']
    v = flow['v']
    mu = flow['mu']
    Reh = flow['Reh']
    dxi = grid['dx']
    deta = grid['dy']

    dudx[1:-1,:] = ((u[2:,:]-u[0:-2,:])/(2*dxi)) #central diff
    dudx[0,:] = ((u[1,:]-u[0,:])/(dxi)) #fwd diff at boundary only
    dudx[-1,:] = ((u[-1,:]-u[-2,:])/(dxi)) #rear diff at boundary only

    dvdx[1:-1,:] = ((v[2:,:]-v[0:-2,:])/(2*dxi)) #central diff
    dvdx[0,:] = ((v[1,:]-v[0,:])/(dxi)) #fwd diff at boundary only
    dvdx[-1,:] = ((v[-1,:]-v[-2,:])/(dxi)) #rear diff at boundary only

    dudy[:,1:] = ((u[:,1:]-u[:,0:-1])/deta)#rear diff
    dudy[:,0] = ((u[:,1]-u[:,0])/deta)# fwd diff at bottom boundary only

    dvdy[:,1:] = ((v[:,1:]-v[:,0:-1])/deta)#rear diff
    dvdy[:,0] = ((v[:,1]-v[:,0])/deta)# fwd diff at bottom boundary only

    tauxy =  (mu/Reh)*(dudy+dvdx)
    tauyy =  ((2*mu)/(3*Reh))*(2*dvdy-dudx)

    return tauxy, tauyy

def viscous_corr_F(grid,flow):
    dudx = np.empty((grid['x_div'], grid['y_div']))
    dudy = np.empty((grid['x_div'], grid['y_div']))
    dvdx = np.empty((grid['x_div'], grid['y_div']))
    dvdy = np.empty((grid['x_div'], grid['y_div']))
    u = flow['u']
    v = flow['v']
    mu = flow['mu']
    Reh = flow['Reh']
    dxi = grid['dx']
    deta = grid['dy']

    dudx[1:-1,:] = ((u[2:,:]-u[0:-2,:])/(2*dxi)) #central diff
    dudx[0,:] = ((u[1,:]-u[0,:])/(dxi)) #fwd diff at boundary only
    dudx[-1,:] = ((u[-1,:]-u[-2,:])/(dxi)) #rear diff at boundary only

    dvdx[1:-1,:] = ((v[2:,:]-v[0:-2,:])/(2*dxi)) #central diff
    dvdx[0,:] = ((v[1,:]-v[0,:])/(dxi)) #fwd diff at boundary only
    dvdx[-1,:] = ((v[-1,:]-v[-2,:])/(dxi)) #rear diff at boundary only

    dudy[:,0:-1] = ((u[:,1:]-u[:,0:-1])/deta)#fwd diff
    dudy[:,-1] = ((u[:,-1]-u[:,-2])/deta)# rear diff at top boundary only

    dvdy[:,0:-1] = ((v[:,1:]-v[:,0:-1])/deta)#fwd diff
    dvdy[:,-1] = ((v[:,-1]-v[:,-2])/deta)# rear diff at top boundary only

    tauxy =  (mu/Reh)*(dudy+dvdx)
    tauyy =  ((2*mu)/(3*Reh))*(2*dvdy-dudx)

    return tauxy, tauyy

def encode_U(flow,grid):
    U = np.empty((grid['x_div'], grid['y_div'], 4))
    U1 = flow['rho']
    U2 = flow['rho']*flow['u']
    U3 = flow['rho']*flow['v']
    U4 = flow['rho']*(flow['e']+(flow['u']**2+flow['v']**2)/2)
    U[:, :, 0] = U1
    U[:, :, 1] = U2
    U[:, :, 2] = U3
    U[:, :, 3] = U4
    return U

def encode_E(flow,qE,tauxx,tauxy_E,grid):
    E = np.empty((grid['x_div'], grid['y_div'], 4))
    rho = flow['rho']
    u = flow['u']
    v = flow['v']
    p = flow['p']
    e = flow['e']
    E1 = rho*u
    E2 = rho*u**2+p-tauxx
    E3 = rho*u*v-tauxy_E
    E4 = (rho*(e+(u**2+v**2)/2)+p)*u+qE-u*tauxx-v*tauxy_E
    E[:, :, 0] = E1
    E[:, :, 1] = E2
    E[:, :, 2] = E3
    E[:, :, 3] = E4
    return E

def encode_F(flow,qF,tauxy_F,tauyy,grid):
    F = np.empty((grid['x_div'], grid['y_div'], 4))
    rho = flow['rho']
    u = flow['u']
    v = flow['v']
    p = flow['p']
    e = flow['e']
    F1 = rho*v
    F2 = rho*u*v-tauxy_F
    F3 = rho*v**2+p-tauyy
    F4 = (rho*(e+(u**2+v**2)/2)+p)*v+qF-u*tauxy_F-v*tauyy
    F[:, :, 0] = F1
    F[:, :, 1] = F2
    F[:, :, 2] = F3
    F[:, :, 3] = F4
    return F

def heat_trans_pred_E(grid,flow):
    dTdx = np.empty((grid['x_div'], grid['y_div']))
    mu = flow['mu']
    M = flow['M']
    Reh = flow['Reh']
    Pr = flow['Pr']
    g = flow['g']
    T = flow['T']
    dx = grid['dx']

    dTdx[1:,:] = ((T[1:,:]-T[0:-1,:])/dx)#rear diff
    dTdx[0,:] = ((T[1,:]-T[0,:])/dx)# fwd diff at left boundary only
    qE = (-mu/((g-1)*M**2*Reh*Pr))*dTdx

    return qE

def heat_trans_corr_E(grid,flow):
    dTdx = np.empty((grid['x_div'], grid['y_div']))
    mu = flow['mu']
    M = flow['M']
    Reh = flow['Reh']
    Pr = flow['Pr']
    g = flow['g']
    T = flow['T']
    dx = grid['dx']

    dTdx[0:-1,:] = ((T[1:,:]-T[0:-1,:])/dx)#fwd diff
    dTdx[-1,:] = ((T[-1,:]-T[-2,:])/dx)# rear diff at right boundary only
    qE = (-mu/((g-1)*M**2*Reh*Pr))*dTdx

    return qE

def heat_trans_pred_F(grid,flow):
    dTdy = np.empty((grid['x_div'], grid['y_div']))
    mu = flow['mu']
    M = flow['M']
    Reh = flow['Reh']
    Pr = flow['Pr']
    g = flow['g']
    T = flow['T']
    dy = grid['dy']

    dTdy[:,1:] = ((T[:,1:]-T[:,0:-1])/dy)#rear diff
    dTdy[:,0] = ((T[:,1]-T[:,0])/dy)# fwd diff at bottom boundary only
    qF = (-mu/((g-1)*M**2*Reh*Pr))*dTdy

    return qF

def heat_trans_corr_F(grid,flow):
    dTdy = np.empty((grid['x_div'], grid['y_div']))
    mu = flow['mu']
    M = flow['M']
    Reh = flow['Reh']
    Pr = flow['Pr']
    g = flow['g']
    T = flow['T']
    dy = grid['dy']

    dTdy[:,0:-1] = ((T[:,1:]-T[:,0:-1])/dy)#fwd diff
    dTdy[:,-1] = ((T[:,-1]-T[:,-2])/dy)# rear diff at top boundary only
    qF = (-mu/((g-1)*M**2*Reh*Pr))*dTdy

    return qF

def decode_U(U,flow):
    U1 = U[1:-1,1:-1,0]
    U2 = U[1:-1,1:-1,1]
    U3 = U[1:-1,1:-1,2]
    U4 = U[1:-1,1:-1,3]
    rho = flow['rho']
    u = flow['u']
    v = flow['v']
    p = flow['p']
    T = flow['T']
    e = flow['e']
    g = flow['g']
    M = flow['M']

    rho[1:-1,1:-1] = U1
    u[1:-1,1:-1] = U2/U1
    v[1:-1,1:-1] = U3/U1
    e[1:-1,1:-1] = U4/U1-0.5*((U2/U1)**2+(U3/U1)**2)
    p[1:-1,1:-1] = (g-1)*(U4-0.5*((U2**2)/U1+(U3**2)/U1))
    T[1:-1,1:-1] = g*M**2*(g-1)*e[1:-1,1:-1]

    flow['rho']=rho
    flow['u']=u
    flow['v']=v
    flow['e']=e
    flow['T']=T
    flow['p']=p

    return flow

def update_BC(grid,flow):
    rho = flow['rho']
    u = flow['u']
    v = flow['v']
    p = flow['p']
    T = flow['T']
    e = flow['e']
    mu = flow['mu']
    Reh = flow['Reh']
    g = flow['g']
    M = flow['M']
    dy = grid['dy']

    #Upper Wall
    u[:,-1] = 1
    v[:,-1] = 0
    p[:,-1] = p[:,-2]+((2*mu[:,-1])/(3*Reh*dy))*(v[:,-1]-2*v[:,-2]+v[:,-3])
    T[:,-1] = 1

    #Lower Wall
    u[:,0] = 0
    v[:,0] = 0
    p[:,0] = p[:,1]-((2*mu[:,0])/(3*Reh*dy))*(v[:,2]-2*v[:,1]+v[:,0])
    T[:,0] = 1

    #Right Outflow Boundary
    u[-1,1:-1] = 2*u[-2,1:-1]-u[-3,1:-1]
    v[-1,1:-1] = 2*v[-2,1:-1]-v[-3,1:-1]
    p[-1,1:-1] = 2*p[-2,1:-1]-p[-3,1:-1]
    T[-1,1:-1] = 2*T[-2,1:-1]-T[-3,1:-1]

    #Left Inflow Boundary
    u[0,1:-1] = 2*u[1,1:-1]-u[2,1:-1]
    v[0,1:-1] = 2*v[1,1:-1]-v[2,1:-1]
    p[0,1:-1] = 2*p[1,1:-1]-p[2,1:-1]
    T[0,1:-1] = 2*T[1,1:-1]-T[2,1:-1]

    #update rho and e
    e = T/(g*M**2*(g-1))
    rho = p/(e*(g-1))

    flow['rho']=rho
    flow['u']=u
    flow['v']=v
    flow['T']=T
    flow['p']=p
    flow['e']=e

    return flow

def update_dynvis(flow):
    mu = flow['mu']
    T = flow['T']
    Tw = 519
    T_dim = T*Tw
    mu = ((T_dim/Tw)**1.5)*((Tw+198.72)/(T_dim+198.72))
    flow['mu'] = mu
    return flow

def convergence(flow,flow_final):
    delta_rho = np.abs(flow_final['rho']-flow['rho'])
    current = 2.375e-3*np.max(delta_rho)
    if current <= 1.0e-8:
        status=True
    else:
        status=False
    return status, current

def save_results(flow,config):
    with open('{0}_rho.dat'.format(config['Name']),'wb') as f:
        np.save(f, flow['rho'])
    with open('{0}_u.dat'.format(config['Name']),'wb') as f:
        np.save(f, flow['u'])
    with open('{0}_v.dat'.format(config['Name']),'wb') as f:
        np.save(f, flow['v'])
    with open('{0}_p.dat'.format(config['Name']),'wb') as f:
        np.save(f, flow['p'])
    with open('{0}_T.dat'.format(config['Name']),'wb') as f:
        np.save(f, flow['T'])
    with open('{0}_e.dat'.format(config['Name']),'wb') as f:
        np.save(f, flow['e'])
    with open('{0}_mu.dat'.format(config['Name']),'wb') as f:
        np.save(f, flow['mu'])
def main():
    configs = read_config('config.csv')

    for config in configs:
        grid, flow  = init_grid(config)
        converged = False
        firstrun = True
        i=0
        while converged==False:
            #calculate timestep
            dt = calc_tstep(grid,flow)
            #if firstrun==True:
            print("The time step is {0}\n".format(dt))
            #    firstrun=False
            flow_pred = mac_predictor(grid,flow,dt)
            pdb.set_trace()
            flow_pred = update_BC(grid,flow_pred)
            pdb.set_trace()
            flow_pred = update_dynvis(flow_pred)
            pdb.set_trace()
            flow_corr = mac_corrector(grid,flow,flow_pred,dt)
            pdb.set_trace()
            flow_corr = update_BC(grid,flow_corr)
            pdb.set_trace()
            flow_corr = update_dynvis(flow_corr)
            pdb.set_trace()
            converged,current = convergence(flow,flow_corr)
            flow = flow_corr
            print("Current max delta_rho is {0}\n".format(current))
            if i >=500:
                i=0
                pdb.set_trace()
                save_results(flow,config)
            else:
                i+=1
        save_results(flow,config)
if __name__ == '__main__':
    main()
