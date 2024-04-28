import numpy as np
import timeit
import csv

def read_config(filename):
    with open(filename) as f:
        dict_reader = csv.DictReader(f, delimiter=',')
    return list(dict_reader)

def init_grid(config):
    x_max = float(config['x_max'])
    y_max = float(config['y_max'])
    x_div = int(config['x_div'])
    y_div = int(config['y_div'])
    Reh = float(config['Reh'])
    g = float(config['g'])
    Pr = float(config['Pr'])

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
    rho = np.ones(x_div,y_div)
    u = np.zeros(x_div,y_div)
    u[:,-1] = np.ones(x_div)
    v = np.zeros(x_div,y_div)
    T = np.full((x_div,y_div),1)
    p = rho*T
    mu = np.ones(x_div,eta_div)

    flow = {'rho':rho,
            'u':u,
            'v':v,
            'T':T,
            'p':p,
            'mu':mu,
            'Reh':Reh,
            'g':g,
            'Pr':Pr
            }
    return grid,flow

def calc_tstep(grid,flow):
    sigma = 0.9
    a = np.sqrt((flow['g']*flow['p'])/flow['rho'])
    dt_CFL = (
            (abs(flow['u'])/grid['dx']) + (abs(flow['v'])/grid['dy'])+
            a*np.sqrt((1/grid['dx']**2) + (1/grid['dy']**2))
            )
    Rex = (flow['rho']*abs(flow['u'])*grid['dx'])/flow['mu']
    Rey = (flow['rho']*abs(flow['v'])*grid['dy'])/flow['mu']

    Re_min = np.min(np.array([Rex Rey]))

    dt = np.min((sigma*dt_CFL)/(1+2/Re_min))
    return dt

def main():
    configs = read_config('config.csv')

    for config in configs:
        grid, flow  = init_grid(config)
        converged = False
        while converged==False:
            #calculate timestep
            tstep = calc_tstep(grid,flow)

if __name__ == '__main__':
    main()
