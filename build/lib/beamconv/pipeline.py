import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import scipy.sparse
from scipy.sparse.linalg import inv

# NaMaster
import pymaster as nmt

# BEAMCONV
from beamconv import ScanStrategy

# DUCC
import ducc0

# PYSM
import pysm3
import pysm3.units as u

# LiteBIRD's IMO
import json
# opening IMO schema.json file and interpreting it as a dictionary
f = open('/afs/mpa/temp/monelli/litebird/litebird_imo-master/IMO/schema.json',)
data = json.load(f)  

# plot-related stuff
import copy
cmap_lighter = copy.copy(matplotlib.cm.get_cmap("YlOrBr_r"))
cmap_viridis = copy.copy(matplotlib.cm.get_cmap("viridis"))
cmap_magma = copy.copy(matplotlib.cm.get_cmap("magma"))

import warnings
warnings.filterwarnings("ignore")

import cycler

import time
start = time.time()

# month of the simulation, might be passed as an input parameter
month = 0

# det 122 is at boresight
first_det = 122
# det 4 has \psi=0, and therefore ensures that I am using the right combination of dets
first_det = 4

########################################################
# INPUT MAPS
########################################################

nside = 128
lmax = 2*nside

sky = pysm3.Sky(nside=nside, preset_strings=["c1"], output_unit="uK_CMB")

nu = 140 # since I'll be using the M1-140 channel

map_FG = sky.get_emission(nu * u.GHz)
alm_FG = hp.map2alm(map_FG, lmax=lmax)

Imin = min(map_FG[0]).value
Imax = max(map_FG[0]).value

Qmin = min(map_FG[1]).value
Qmax = max(map_FG[1]).value

Umin = min(map_FG[2]).value
Umax = max(map_FG[2]).value

hp.visufunc.mollview(map_FG[0],title='I input', min=Imin, max=Imax)
plt.savefig('I-1.png')

hp.visufunc.mollview(map_FG[1],title='Q input', min=Qmin, max=Qmax)
plt.savefig('Q-1.png')

hp.visufunc.mollview(map_FG[2],title='U input', min=Umin, max=Umax)
plt.savefig('U-1.png')

########################################################
# SCANNING STRATEGY
########################################################

sampling_freq = 19.0

# setting up the scanning strategy parameters
ctime0 = 1510000000            # initial time, might be passed as input parameter
mlen = 10 * 24 * 60 * 60       # mission length in seconds (one month!)

# Definition of the scanning strategy making use of LiteBIRD's specifics (with HWP non-idealities)
ss = ScanStrategy(
        duration = mlen,
        external_pointing = True,
        theta_antisun = 45., 	         # [deg]
        theta_boresight = 50.,           # [deg]
        freq_antisun = 192.348,          # [min]
        freq_boresight = 0.314,          # [rad/min]
        sample_rate = sampling_freq,     # [Hz]
        jitter_amp = 0.0,
        ctime0 = ctime0)

# Further options (non-ideal HWP)
scan_opts = dict(
        q_bore_func = ss.litebird_scan,
        ctime_func = ss.litebird_ctime,
        use_litebird_scan = True,
        q_bore_kwargs = dict(),
        ctime_kwargs = dict(),
        max_spin = 2,
        nside_spin = nside,
        preview_pointing = False,
        verbose = True,
        save_tod = True,
        save_point = True)

nchunk = 10

# calculate TOD in 10 chunks
nsamp_chunk = int(ss.mlen * ss.fsamp / nchunk)
nsamp = int(ss.mlen * ss.fsamp)
chunks = ss.partition_mission(nsamp_chunk)

########################################################
# HWP SPECIFICS
########################################################

bandcenter = nu
bandwidth = 42.0

trans_rec = np.genfromtxt('MFT_HWP.csv', delimiter=';')

####################################################################
# Alex's definitions!

jones = np.zeros((trans_rec.shape[0],2,2), dtype='complex128')

jones[:,0,0] = trans_rec[:,1]**0.5
jones[:,0,1] = 10**(trans_rec[:,6]/10.)
jones[:,1,0] = 10**(trans_rec[:,6]/10.)
jones[:,1,1] = -trans_rec[:,3]**0.5*np.exp(1j*np.radians(trans_rec[:,5]-180))

####################################################################
# From Jones to Mueller matrix elements

###From Tom Hileman's transfer_matrix.py
Sigma = [] # Pauli + Indentity
Sigma.append( np.array(( (1,0),(0,1)), dtype=complex)) # identity matrix
Sigma.append( np.array(( (1,0),(0,-1)), dtype=complex))
Sigma.append( np.array(( (0,1),(1,0)), dtype=complex))
Sigma.append( np.array(( (0,-1j),(1j,0)), dtype=complex)) # Need to multiply by -1 to change back to normal.

mueller = np.zeros((trans_rec.shape[0],4,4))

# is this the correct definition?
freqs_array = np.concatenate(np.where(np.logical_and(trans_rec[:,0]>=bandcenter-bandwidth/2, trans_rec[:,0]<=bandcenter+bandwidth/2)))
    
for i in freqs_array:
    for j in range(4):
        for k in range(4):
            temp = .5*np.trace( np.dot(Sigma[j], np.dot(jones[i], np.dot(Sigma[k], jones[i].conj().transpose()))))
            mueller[i,j,k] = np.real(temp)

index_140GHz = np.where(trans_rec[:,0] == bandcenter)
mueller_140GHz = mueller[index_140GHz,:,:].reshape((4,4))

print('mueller matrix that will be used:')
print(mueller_140GHz)

########################################################
# FOCAL PLANE SPECIFICS
########################################################

# looking into the IMO, data['data_files'] is where the relevant info is stored 
data_files = data['data_files']

# counting how many objects are in data_files
nkey=0
for key in data_files:
    nkey = nkey+1

# looking for the detectors belonging to the M1-140 channel
for i in range(nkey):
    test = data_files[i]
    if(test['name'] == 'channel_info'):
        metadata = test['metadata']
        if(metadata['channel'] == 'M1-140'):
            detector_names = metadata['detector_names']
            break

ndet = len(detector_names)
det_indices = range(ndet)

list_of_dictionaries = []

# looking for the metadata of the detectors in detector_names
for d in detector_names:
    for j in range(nkey):
        test = data_files[j]
        if(test['name'] == 'detector_info'):
            metadata = test['metadata']
            if (metadata['name'] == d):
                list_of_dictionaries.append(metadata) # this list contains all the info about the M1-140 detectors
                break
            
# the following quantities are actually identical for each detector
# (I checked that by means of the all_equals function defined below)
fwhm = list_of_dictionaries[0]['fwhm_arcmin']
ellipticity = list_of_dictionaries[0]['ellipticity'] # it's zero, IS THIS REALISTIC?
bandcenter = list_of_dictionaries[0]['bandcenter_ghz']
bandwidth = list_of_dictionaries[0]['bandwidth_ghz']
sampling_freq = list_of_dictionaries[0]['sampling_rate_hz']
net = list_of_dictionaries[0]['net_ukrts']
pol_sensitivity = list_of_dictionaries[0]['pol_sensitivity_ukarcmin']
fknee = list_of_dictionaries[0]['fknee_mhz']
fmin = list_of_dictionaries[0]['fmin_hz']
alpha = list_of_dictionaries[0]['alpha']

def all_equals(string):
    test = np.empty(ndet,dtype=object)
    for i in det_indices:
        test[i] = list_of_dictionaries[i][string]
    return all(x==test[0] for x in test)
# example of usage: print(all_equals('alpha'))

# instead, the following change detector by detector
pol_array = np.empty(ndet,dtype=object)
orient_array = np.empty(ndet,dtype=object)
quat_array = np.empty((ndet,4))

for i in det_indices:
    pol_array[i] = list_of_dictionaries[i]['pol']
    orient_array[i] = list_of_dictionaries[i]['orient']
    quat_array[i] = np.array(list_of_dictionaries[i]['quat'])

# this was to check which det is at boresight    
# for i in det_indices:
#     if list_of_dictionaries[i]['name']=='M03_030_QA_140T':
#         print(i)
# for i in det_indices:
#     if list_of_dictionaries[i]['name']=='M03_030_QA_140B':
#         print(i)
        
# this was to check that det 122 is actually at boresight        
# print('detector '+str(list_of_dictionaries[122]['name'])+': '+str(quat_array[122]))
# print('detector '+str(list_of_dictionaries[123]['name'])+': '+str(quat_array[123]))

# create a grid of Gaussian beams
# ndet was already defined! might be worth to call it differetly
ndet = 4

azs = np.zeros((ndet,2))
els = np.zeros((ndet,2))
polangs = np.zeros((ndet,2))
quats = np.zeros((ndet,2,4))
deads = np.tile(np.array([0,1]),(ndet,1))

for i in range(ndet):
    quats[i,0,0] = quat_array[first_det+i,3] # the offset quaternions are taken from the IMO
    quats[i,0,1:4] = quat_array[first_det+i,0:3] # the offset quaternions are taken from the IMO
    if list_of_dictionaries[first_det+i]['orient']=='Q':
        if list_of_dictionaries[first_det+i]['pol']=='T':
            polangs[i,0] = 0
        else: polangs[i,0] = 90
    else:
        if list_of_dictionaries[first_det+i]['pol']=='T':
            polangs[i,0] = 45
        else: polangs[i,0] = 135

print('the detectors I am using have \psi:')
print(polangs)
    
# setting up the beam options
beam_opts = dict(lmax=lmax,
                 btype='Gaussian',
                 fwhm=fwhm,          # gaussian co-pol beam, so only specify FWHM (arcmin)
                 hwp_mueller=mueller_140GHz,
                 #hwp_mueller=np.diag([1,1,-1,-1]),
                 #hwp_mueller=np.diag([1,1,1,1]),
                 quats=quats
                )

# defining HWP frequency
ss.set_hwp_mod(mode='continuous', freq=88/60)
#ss.set_hwp_mod(mode='continuous', freq=0)

# creating the focal plane
ss.input_focal_plane(azs, els, polangs, deads, combine=True, scatter=False, **beam_opts)

# Producing the coverage map
ss.allocate_maps(nside=128)
ss.scan_instrument_mpi(alm_FG, **scan_opts)
maps, cond, proj = ss.solve_for_map(return_proj = True)

hp.mollview(maps[0], title="I", min=Imin, max=Imax)
plt.savefig('I-2.png')

hp.mollview(maps[1], title="Q", min=Qmin, max=Qmax)
plt.savefig('Q-2.png')

hp.mollview(maps[2], title="U", min=Umin, max=Umax)
plt.savefig('U-2.png')

#######################################

mask_raw = proj

# The following function calls create apodized versions of the raw mask
# with an apodization scale of 2.5 degrees using three different methods

# Apodization scale in degrees
aposcale = 2.5

# C1 and C2: in these cases, pixels are multiplied by a factor f
#            (with 0<=f<=1) based on their distance to the nearest fully
#            masked pixel. The choices of f in each case are documented in
#            Section 3.4 of the C API documentation. All pixels separated
#            from any masked pixel by more than the apodization scale are
#            left untouched.
mask_C1 = nmt.mask_apodization(mask_raw, aposcale, apotype="C1")
mask_C2 = nmt.mask_apodization(mask_raw, aposcale, apotype="C2")

# Smooth: in this case, all pixels closer to a masked pixel than 2.5 times
#         the apodization scale are initially set to zero. The resulting
#         map is then smoothed with a Gaussian kernel with standard
#         deviation given by the apodization scale. Finally, all pixels
#         originally masked are forced back to zero.
mask_Sm = nmt.mask_apodization(mask_raw, aposcale, apotype="Smooth")

hp.mollview(mask_Sm, coord=['G', 'C'], title='Apodized mask')
plt.show()

# Read healpix maps and initialize a spin-0 and spin-2 field
f_0 = nmt.NmtField(mask, [hp.read_map("maps.fits", field=0, verbose=False)])
f_2 = nmt.NmtField(mask, hp.read_map("maps.fits", field=[1, 2], verbose=False))

# Initialize binning scheme with 4 ells per bandpower
b = nmt.NmtBin.from_nside_linear(nside, 4)

# Compute MASTER estimator
# spin-0 x spin-0
cl_00 = nmt.compute_full_master(f_0, f_0, b)
# spin-0 x spin-2
cl_02 = nmt.compute_full_master(f_0, f_2, b)
# spin-2 x spin-2
cl_22 = nmt.compute_full_master(f_2, f_2, b)

# Plot results
ell_arr = b.get_effective_ells()
plt.plot(ell_arr, cl_00[0], 'r-', label='TT')
plt.plot(ell_arr, np.fabs(cl_02[0]), 'g-', label='TE')
plt.plot(ell_arr, cl_22[0], 'b-', label='EE')
plt.plot(ell_arr, cl_22[3], 'y-', label='BB')
plt.loglog()
plt.xlabel('$\\ell$', fontsize=16)
plt.ylabel('$C_\\ell$', fontsize=16)
plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
plt.show()


Cl_input = hp.anafast(map_FG, lmax=lmax)
Cl_proce = hp.anafast(maps, lmax=lmax)
ell = np.arange(lmax+1)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
fig.suptitle('Sharing x per column, y per row')
ax1.plot(ell, ell * (ell + 1) * Cl_input[0])
ax1.plot(ell, ell * (ell + 1) * Cl_proce[0])
ax2.plot(ell, ell * (ell + 1) * Cl_input[1])
ax2.plot(ell, ell * (ell + 1) * Cl_proce[1])
ax3.plot(ell, ell * (ell + 1) * Cl_input[2])
ax3.plot(ell, ell * (ell + 1) * Cl_proce[2])
ax4.plot(ell, ell * (ell + 1) * Cl_input[3])
ax4.plot(ell, ell * (ell + 1) * Cl_proce[3])
ax5.plot(ell, ell * (ell + 1) * Cl_input[4])
ax5.plot(ell, ell * (ell + 1) * Cl_proce[4])
ax6.plot(ell, ell * (ell + 1) * Cl_input[5])
ax6.plot(ell, ell * (ell + 1) * Cl_proce[5])

for ax in fig.get_axes():
    ax.label_outer()
    
plt.savefig('Cls.png')

quit()

# Setting up and filling TOD (noiseless)
psi = np.empty((ndet,nsamp))
pix = np.empty((ndet,nsamp))
noiseless_TOD = np.empty((ndet,nsamp))
phi = np.empty(nsamp)

for d in range(ndet):
    for chunk in range(nchunk):
        psi[d,chunk*nsamp_chunk:(chunk+1)*nsamp_chunk] = ss.data(chunks[chunk],ss.beams[d][0],data_type='pa')
        pix[d,chunk*nsamp_chunk:(chunk+1)*nsamp_chunk] = ss.data(chunks[chunk],ss.beams[d][0],data_type='pix')
        #
        noiseless_data = ss.data(chunks[chunk],ss.beams[d][0],data_type='tod')
        noiseless_TOD[d,chunk*nsamp_chunk:(chunk+1)*nsamp_chunk] = noiseless_data
        #
        phi[chunk*nsamp_chunk:(chunk+1)*nsamp_chunk] = ss.data(chunks[chunk],data_type='hwp_ang')
        
now = time.time()
delta = now - start
print('TOD produced in ' + str(delta) + ' seconds')

# variables should be unified! this is just redundant
nobs = nsamp
dets = ndet

pixel_new = np.empty((ndet,nsamp))

pix_array = pix.reshape(ndet*nsamp)

# I checked that pix_array was defined correctly, it seemed so
# print('pix:')
# print(pix)
#
# print('pix_array:')
# print(pix_array)

pix_reduced = np.array(sorted(list(set(pix_array))),dtype='int32')
integers = np.arange(0,len(pix_reduced),dtype='int32')
dic = dict(zip(pix_reduced,integers))

nhits = len(pix_reduced)
print(str(nhits) + ' pixels observed')

for d in range(ndet):
    pixel_new[d] = np.vectorize(dic.get)(pix_array[d*nobs:(d+1)*nobs])

# old checks:
# print(min(pixel_new[d]))
# print(max(pixel_new[d]))
#    
# print(np.shape(pixel_new))
# print(pixel_new[0])

row = np.zeros(3*nobs*dets, dtype=np.int32)
column = np.zeros(3*nobs*dets, dtype=np.int32)
val = np.zeros(3*nobs*dets, dtype=np.float64)

##########################################################
# ARE THE FOLLOWING DEFINITION COMPATIBLE WITH BEAMCONV?
##########################################################
#omega = 88*2*np.pi/60 #so that it is in rad/s
#phi0 = 0 #initial HWP angle

#def phi(t):
#    return phi0 + omega*t

#def alpha(t,psi):
#    return 2*phi(t) + 2*psi

def alpha(phi,psi):
    return 2*phi + 2*psi
 
def beta(phi,xi):
    return 2*phi - 2*xi
    
# fin qui sta in piedi
# vanno definite tutte le robe che compaiono qui sotto!

idx_reduced = np.arange(0,nobs)
t_sec = month*10*24*60*60 + idx_reduced/ss.fsamp

mII = mueller_140GHz[0,0]
mIQ = mueller_140GHz[0,1]
mIU = mueller_140GHz[0,2]
mQI = mueller_140GHz[1,0]
mQQ = mueller_140GHz[1,1]
mQU = mueller_140GHz[1,2]
mUI = mueller_140GHz[2,0]
mUQ = mueller_140GHz[2,1]
mUU = mueller_140GHz[2,2]

#mII = 1
#mIQ = 0
#mIU = 0
#mQI = 0
#mQQ = 1
#mQU = 0
#mUI = 0
#mUQ = 0
#mUU = -1

#mII = 1
#mIQ = 0
#mIU = 0
#mQI = 0
#mQQ = 1
#mQU = 0
#mUI = 0
#mUQ = 0
#mUU = 1

xi_array = polangs[:,0]/360*2*np.pi #so that it is in rad

d = np.zeros(nobs*dets)

for j in range(ndet):
    Psi = psi[j]/360*2*np.pi #so that it is in rad
    
    row[j*3*nobs:(j+1)*3*nobs:3] = j*nobs+idx_reduced
    column[j*3*nobs:(j+1)*3*nobs:3] = 3*pixel_new[j]
    val[j*3*nobs:(j+1)*3*nobs:3] = (mII+mQI*np.cos(beta(phi,xi_array[j]))-mUI*np.sin(beta(phi,xi_array[j])))/2
    
    row[j*3*nobs+1:(j+1)*3*nobs:3] = j*nobs+idx_reduced
    column[j*3*nobs+1:(j+1)*3*nobs:3] = 3*pixel_new[j]+1
    val[j*3*nobs+1:(j+1)*3*nobs:3] = (mIQ*np.cos(alpha(phi,Psi))-mIU*np.sin(alpha(phi,Psi))+(mQQ*np.cos(alpha(phi,Psi))-mQU*np.sin(alpha(phi,Psi)))*np.cos(beta(phi,xi_array[j]))-(mUQ*np.cos(alpha(phi,Psi))-mUU*np.sin(alpha(phi,Psi)))*np.sin(beta(phi,xi_array[j])))/2
    
    row[j*3*nobs+2:(j+1)*3*nobs:3] = j*nobs+idx_reduced
    column[j*3*nobs+2:(j+1)*3*nobs:3] = 3*pixel_new[j]+2
    val[j*3*nobs+2:(j+1)*3*nobs:3] = (mIQ*np.sin(alpha(phi,Psi))+mIU*np.cos(alpha(phi,Psi))+(mQQ*np.sin(alpha(phi,Psi))+mQU*np.cos(alpha(phi,Psi)))*np.cos(beta(phi,xi_array[j]))-(mUQ*np.sin(alpha(phi,Psi))+mUU*np.cos(alpha(phi,Psi)))*np.sin(beta(phi,xi_array[j])))/2 
    
    d[j*nobs:(j+1)*nobs] = noiseless_TOD[j]
    
mat = scipy.sparse.coo_matrix((val,(row,column)),shape=(dets*nobs,3*nhits)).tocsr()
del row, column, val

print('matrix done: '+str(time.time()-start))

ATA = mat.transpose().dot(mat)
ATA_block = np.zeros((3,3))

# index to count singular blocks
sing = 0

# list of "singular" pixels
sing_pix = []

row = np.zeros(9*nhits, dtype=np.int32)
column = np.zeros(9*nhits, dtype=np.int32)
val = np.zeros(9*nhits, dtype=np.float64)

for p in range(nhits):
    ATA_block[0,0] = ATA[3*p,3*p]
    ATA_block[0,1] = ATA[3*p,3*p+1]
    ATA_block[0,2] = ATA[3*p,3*p+2]
    ATA_block[1,0] = ATA[3*p+1,3*p]
    ATA_block[1,1] = ATA[3*p+1,3*p+1]
    ATA_block[1,2] = ATA[3*p+1,3*p+2]
    ATA_block[2,0] = ATA[3*p+2,3*p]
    ATA_block[2,1] = ATA[3*p+2,3*p+1]
    ATA_block[2,2] = ATA[3*p+2,3*p+2]

    if np.linalg.det(ATA_block):
        ATA_inv = np.linalg.inv(ATA_block)
        #
        row[9*p:9*p+3] = 3*p
        row[9*p+3:9*p+6] = 3*p+1
        row[9*p+6:9*(p+1)] = 3*p+2
        column[9*p:9*(p+1):3] = 3*p
        column[9*p+1:9*(p+1):3] = 3*p+1
        column[9*p+2:9*(p+1):3] = 3*p+2
        val[9*p:9*(p+1)] = ATA_inv.flatten()
    else:
        sing += 1
        sing_pix.append(p)

print('fraction of singular blocks: ' + str(sing) + '/' + str(nhits))
        
inverse = scipy.sparse.coo_matrix((val,(row,column)),shape=(3*nhits,3*nhits)).tocsr()
del row, column, val
    
print('inversion completed: ' + str(time.time()-start))

s = inverse.dot(mat.transpose()).dot(d)

# does this help?
for p in sing_pix:
    s[3*p:3*(p+1)] = hp.pixelfunc.UNSEEN

print('map done: '+str(time.time()-start))

npix = hp.nside2npix(128)

I_signal = np.full(npix,hp.pixelfunc.UNSEEN)
Q_signal = np.full(npix,hp.pixelfunc.UNSEEN)
U_signal = np.full(npix,hp.pixelfunc.UNSEEN)

I_signal[pix_reduced] = s[3*integers]
Q_signal[pix_reduced] = s[3*integers+1]
U_signal[pix_reduced] = s[3*integers+2]

#col_00 = fits.Column(name='pixel', format='1D', unit='integer', array=pixel_array)
#col_01 = fits.Column(name='I', format='1D', unit='Stokes I', array=I_signal)
#col_02 = fits.Column(name='Q', format='1D', unit='Stokes Q', array=Q_signal)
#col_03 = fits.Column(name='U', format='1D', unit='Stokes U', array=U_signal)

#cols = fits.ColDefs([col_00, col_01, col_02, col_03])
#hdr_1 = fits.Header()
#hdr_1['NSIDE'] = nside
#hdu_1 = fits.BinTableHDU.from_columns(cols, header=hdr_1)

#hdr = fits.Header()
#hdr['Author'] = 'Marta'
#primary_hdu = fits.PrimaryHDU(header=hdr)

#hdul = fits.HDUList([primary_hdu,hdu_1])
#hdul.writeto('output_maps/beam'+beam_string+'_nside'+nside_string+'/monthly_both_'+month_string+'.fits')

#print(month_string+' finished! ('+str(time.time()-start)+')')

hp.mollview(I_signal, title="I reconstructed", min=Imin, max=Imax)
plt.savefig('I-2.png')

hp.mollview(Q_signal, title="Q reconstructed", min=Qmin, max=Qmax)
plt.savefig('Q-2.png')

hp.mollview(U_signal, title="U reconstructed", min=Umin, max=Umax)
plt.savefig('U-2.png')

I_signal2 = np.full(npix,hp.pixelfunc.UNSEEN)
Q_signal2 = np.full(npix,hp.pixelfunc.UNSEEN)
U_signal2 = np.full(npix,hp.pixelfunc.UNSEEN)

for p in pix_reduced:
    I_signal2[p] = I_signal[p]/(map_FG[0,p].value)
    Q_signal2[p] = Q_signal[p]/(map_FG[1,p].value)
    U_signal2[p] = U_signal[p]/(map_FG[2,p].value)


hp.mollview(I_signal2, title="I")
plt.savefig('I.png')

hp.mollview(Q_signal2, title="Q")
plt.savefig('Q.png')

hp.mollview(U_signal2, title="U")
plt.savefig('U.png')