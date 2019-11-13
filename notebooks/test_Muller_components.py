import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import copy
import pickle
import warnings
import numpy as np
import healpy as hp
import argparse as ap
from beamconv.instrument import ScanStrategy, Beam
from beamconv import tools as beam_tools
from beamconv import plot_tools
import scipy.constants as constants
import emcee
from mpi4py import MPI

comm = MPI.COMM_WORLD
c=constants.c

def log_prob(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)

def compute_best_value(frequency):
	'''
	compute the best value for the sapphire thicknesses for each frequency)

	Arguments:
		frequency : Hz
		n1,n2 : sapphire fast and slow indices
	'''
	bv = c/(2*frequency*np.abs(n1-n2))
	
	return bv

def set_values():
	'''
	assign the allowed values range for the parameters
	'''

	freq_v = np.arange(30, 465, 15)
	ind_v = np.arange(1.5, 3.1, 0.1)
	thick_d_v = np.arange(0, 2.51e-03, 0.01)
	thick_s_v = np.arange(-bv, bv+0.01, 0.01)
	in_angle_v = np.arange(0, 21, 1)

	return freq_v, ind_v, thick_d_v, thick_s_v, in_angle_v

def test_hwp_models(model, freq, ind, thick, in_angle,  
	losses, fixed_values=True, check=False, estimator='abs_diff', plot_stuff=False):
	'''
	Test different hwp structures for fixed or fluctuating parameters

	Arguments:
		model : the model of the hwp
		freq : sensitive frequency
		ind : indices of the coatings
		thick_d : duroid thicknesses
		thick_s : sapphire thicknesses (optimized for each frequency through compute_best_val)
		in_angle : angle of incidence

		'''
	if not fixed_values:
		# not ready, needs alterations
		#	ndim, nwalkers = 5, 100
		#	ivar = 1. / np.random.rand(ndim)
		#	p0 = np.random.randn(nwalkers, ndim)
		#	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
		#	sampler.run_mcmc(p0, 10000)
		pass

	if check:
		freq_v, ind_v, thick_d_v, thick_s_v, in_angle_v = set_values()
		if ([freq,ind,thick_d,thick_s,in_angle,layers] not in 
			zip(freq_v, ind_v, thick_d_v, thick_s_v, in_angle_v)):

			raise ValueError('Parameter values are incorrect')

	test_hwp = Beam().hwp()
	if model:
		test_hwp.choose_HWP_model(model)
	else:
		test_hwp.stack_builder(thicknesses=thick, indices=ind, 
							losses=losses, angles=np.array([0.0, 0.0, 0.0]))
	T,rho,c,s = test_hwp.compute4params(freq,in_angle)

	if estimator == 'abs_diff':
		F = (T-1)**2 + rho**2 + (c+1)**2 + s**2
		print(F)

	#if args.system_name == 'Owl':

	#	base_dir = '/mn/stornext/d8/ITA/spider/jon/analysis/'
	#	base_dir_adri = '/mn/stornext/d8/ITA/spider/adri/analysis/'
	#	outdir = opj(base_dir, '20191108_hwpsat_test/')
	#	beam_dirs = opj(base_dir_adri, '20180627_sat/beams')

    #elif args.system_name == 'nadia':

	#base_dir = '/home/nadia/git_repos/hwp_sims/'
	#outdir = opj(base_dir, 'hwp_script_out/')

	#np.save(opj(outdir,'abs_diff.npy'), F)



def main():

	test_hwp_models('1layer_HWP', 150e09, np.array([[1.5,1.5],[3.3,3.1],[1.5,1.5]]),np.array([0.2e-3, 4e-3,0.2e-3]), 0.01, np.ones((3,2))*1e-03,)

if __name__ == '__main__':
    main()