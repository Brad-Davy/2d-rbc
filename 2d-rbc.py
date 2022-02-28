"""Dedalus simulation of 2d Rayleigh benard convection.


   This script solves the temperature equation and applies periodic boundary conditions in the horizontal
   direction. The top and bottom boundaries are set to no slip.

Usage:
    2d-rbc.py --ra=<rayleigh> --N=<resolution> --max_dt=<maximum_dt> --init_dt=<Initial_dt> [--pr=<prandtl>] [--mesh=<mesh>]
    2d-rbc.py -h | --help

Options:
    -h --help   Display this help message
    --ra=<rayliegh>     Rayleigh number
    --N=<resolution>    Nx=2Nz
    --max_dt=<max_dt>   Maximum Time step
    --init_dt=<init_dt> Initial Time step
    --pr=<prandtl>      Prandtl number [default: 7]
    --mesh=<mesh>       Parallel mesh [default: None]
"""
## Import libraries ##
import numpy as np
from mpi4py import MPI
import time
from docopt import docopt
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
import os
logger = logging.getLogger(__name__)

## Docopt ##
args=docopt(__doc__)
comm = MPI.COMM_WORLD

## Parameters ##
Lx, Lz = (2, 1.)


## Define the number of spectral modes ##
N = int(args['--N'])
Nx = N
Nz = int(N/2)
max_dt = float(args['--max_dt'])

## Take the input arguments from Docopt ##
Rayleigh = float(args['--ra'])
Prandtl = float(args['--pr'])

## Create the file tag to make a unique directory ##
file_tag="Ra_{:.2e}_Pr_{}_N_{}_".format(Rayleigh,Prandtl,N)
file_tag=file_tag.replace(".","-")

if comm.rank==0:
    os.system('mkdir results/{}'.format(file_tag))

if args['--mesh']!="None":
    mesh = (int(args['--mesh'].split(',')[0]),int(args['--mesh'].split(',')[1]))
else:
    mesh = None


## Create bases and domain ##
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

## 3D Boussinesq hydrodynamics ##
problem = de.IVP(domain, variables=['p','Θ','u','w','Θz','uz','wz'], time='t')
problem.parameters['Pr'] = Prandtl
problem.parameters['Ra'] = Rayleigh
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz

## Governing equations ##
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(Θ) - (dx(dx(Θ)) + dz(Θz))                        = - u*dx(Θ) - w*Θz")
problem.add_equation("dt(u) - Pr*(dx(dx(u)) + dz(uz)) + dx(p) = - u*dx(u) - u*dx(u) - w*uz")
problem.add_equation("dt(w) - Pr*(dx(dx(w)) + dz(wz)) + dz(p) - Pr*Ra*Θ   = - u*dx(w) - w*wz")

problem.add_equation("Θz - dz(Θ) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")

## Boundary conditions ##
problem.add_bc("left(Θ) = 1")
problem.add_bc("right(Θ) = 0")
problem.add_bc("left(u)= 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("integ_z(p) = 0", condition="(nx == 0)")


## Build solver ##
solver = problem.build_solver(de.timesteppers.SBDF2)

## Initial conditions ##
z = domain.grid(1)
Θ = solver.state['Θ']
u = solver.state['u']


## Random perturbations, initialized globally for same results in parallel ##
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]
zb, zt = z_basis.interval
pert =  1e-6 * noise * (zt - z) * (z - zb)
Θ['g'] = 0.1*(z - pert)

## Integration parameters ##
sim_end_time=5
max_iterations=100000
sim_wall_time=24*60*60


## Snapshots of the entire domain, only saves 10 for memory ##
snap = solver.evaluator.add_file_handler('results/{}/{}snapshots'.format(file_tag, file_tag), sim_dt=0.02, max_writes = 10)
snap.add_system(solver.state)


## Dedalus analysis, saved in the h5 format ##
analysis = solver.evaluator.add_file_handler("results/{}/{}analysis".format(file_tag,file_tag),iter=5, max_writes=np.inf)
analysis.add_task("interp((1/Lx)*integ( -dz(Θ) + w*Θ ,'x'),z=0.5)", name = 'nu_mid_plane')
analysis.add_task("(1/Lz)*integ((1/Lx)*integ( -dz(Θ) + w*Θ ,'x'),'z')", name = 'nu_integral')
analysis.add_task("interp( (1/Lx)*integ(-dz(Θ) ,'x') , z=0)", name='nu_bot_wall')
analysis.add_task("interp((1/Lx)*integ(-dz(Θ) ,'x') , z=1)", name='nu_top_wall')
analysis.add_task("(1/Lx)*integ(Θ,'x')", name="Θ_prof")
analysis.add_task("(1/Lx)*integ(sqrt(u*u),'x')", name="u_prof")
analysis.add_task("(1/Lx)*integ(sqrt(w*w),'x')", name="w_prof")
analysis.add_task("(1/Lx)*integ( Θz,'x')", name="conduction_prof")
analysis.add_task("(1/Lx)*integ( w*Θ,'x')", name="advection_prof")

## Peclet and Raynolds number calculations ##
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(1/(Lx*Lz))*(integ(u*u + w*w))**0.5", name = 'Peclet')
flow.add_property("(1/Pr)*(1/(Lx*Lz))*(integ(u*u + w*w))**0.5",name = 'Reynolds')

## Nusselt number calculations ##
flow.add_property("interp((1/Lx)*integ( -dz(Θ) + w*Θ,'x'), z = 0.5)",name = 'nu_mid_plane')
flow.add_property("(1/(Lx*Lz))*integ(w*Θ - Θz)", name = 'Integral_Nusselt')
flow.add_property("interp((1/Lx)*integ(-dz(Θ) ,'x'), z = 0)",name='nu_bot_wall')
flow.add_property("interp((1/Lx)*integ(-dz(Θ) ,'x'), z = 1)",name='nu_top_wall')

## Properties of the flow ##
flow.add_property('u',name='u')
flow.add_property('w',name='w')
flow.add_property('p', name='pressure')
flow.add_property('Θ',name = 'temperature')


for field in solver.state.fields:
    field.set_scales(domain.dealias)

## Write to the output file ##
log_file = open('results/{}/{}log.txt'.format(file_tag,file_tag),'w')
log_file.write('\n Time			Nusselt			Peclet')
log_file.close()	


## CFL ##
CFL = flow_tools.CFL(solver, initial_dt = 1e-7, cadence=5, safety=0.5, max_change=1.5, min_change=0.5, max_dt = max_dt)
CFL.add_velocities(('u','w'))



## Main loop ##
try:
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
                uu = flow.max('u')
                ww = flow.max('w')
                nu_integrated = flow.max('Integral_Nusselt')
                nu_bot = flow.max('nu_bot_wall')
                nu_top = flow.max('nu_top_wall')
                pe = flow.max('Peclet')
                logger.info(" ")
                logger.info("-"*60)
                logger.info("Ra:{:.4e}, Pr:{:.1f}, N:{}".format(Rayleigh, Prandtl, N))
                logger.info("-"*60)
                logger.info('u:{:.4e}, w:{:.4e}'.format(uu,ww))
       	        logger.info('Iteration: {:.0f}'.format(solver.iteration))
                logger.info('Integral Nu = {:.4f}, top Nu = {:.4f}, bottom Nu = {:.4f}'.format(nu_integrated,nu_top,nu_bot))
                logger.info('The Peclet number is {:.3f}'.format(pe))
                logger.info('The current time is: {:.3f}'.format(float(solver.sim_time)))
                logger.info('The currently using {:.2f}% of time step.'.format(float((dt/max_dt)*100)))
                
                if comm.rank == 0:
                     log_file = open('results/{}/{}log.txt'.format(file_tag,file_tag),'a')
                     log_file.write('\n{:.4e}		{:.3f} 		     	{:.3f}'.format(float(solver.sim_time),nu_integrated, pe))
                     log_file.close()
except:  
    pass
finally:
    end_run_time = time.time()
