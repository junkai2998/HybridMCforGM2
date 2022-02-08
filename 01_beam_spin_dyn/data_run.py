import argparse

################### Command Line Interface ##################
parser = argparse.ArgumentParser(description='generate data.')
parser.add_argument('-N', type=int,default=10,help='number of runs default = 10')

args = parser.parse_args()

Nrun = args.N

print("Current run: ",Nrun)
###########################################################

import sys, os, shutil, time, datetime
# import ROOT as r
import numpy as np
# import matplotlib.pyplot as plt

sys.path.insert(1,'../02_positron_sampling')
from PositronDecay import GenerateMichelKinematics, GeneratePositron_MRF, Calculate_Phase, GeneratePositron_LAB
from Constants import EMMU, EMASS, pi, twopi, fine_structure_const



N_per_fill = 10000000
get_theta_c = lambda t:2*np.pi/0.1492*t # cyclotron motion, phase set to 0
get_theta_s = lambda t:2*np.pi/0.1443*t # spin precession, phase set to 0

def GenerateFill(j):
    # with time randomization
    print ("Fill: ",j) # to use a function for multiprocessing.pool, function has to take 1 argument
    
    four_momenta_LAB = np.zeros(shape=(N_per_fill,8))
    
    t_pts = 149.2*np.random.random(size=N_per_fill) # continuous time in 700 us
    cyclotron_angles = get_theta_c(t_pts)
    spin_precession_angles = get_theta_s(t_pts)

    four_momenta_LAB[:,7] = t_pts # posiInitTime

    for i in range(N_per_fill):
        # positrons in MRF
        theta_s = spin_precession_angles[i]
        E_primed, px_primed, py_primed, pz_primed, muDecayPolX, muDecayPolY, muDecayPolZ = GeneratePositron_MRF(*GenerateMichelKinematics(),theta_s)
        
        # boost to LAB frame
        theta_c = cyclotron_angles[i]
        four_momenta_LAB[i,0:7] = GeneratePositron_LAB(E_primed, px_primed, py_primed, pz_primed, theta_c)
    
#     return four_momenta_LAB
    np.save('data_run/run_{}_{}.npy'.format(Nrun,j),four_momenta_LAB)
    print ("data_run/run_{}_{}.npy' saved".format(Nrun,j))


from multiprocessing import Pool,cpu_count
N_cpu_selected = 10 # overide auto choosing
N_fills = 10 # number of fills that you want to produce

print("start processing data")
start_time = time.time()
# parallelization
pool = Pool(processes=N_cpu_selected)
fills = pool.map(GenerateFill, range(N_fills))
pool.close()
pool.join()
print("--- total time taken: %s ---" % (str(datetime.timedelta(seconds=time.time() - start_time))))

# collect all fills
# four_momenta_LAB = np.vstack(fills)

# np.save('20Mxz_y0_tcontinuous.npy',four_momenta_LAB)