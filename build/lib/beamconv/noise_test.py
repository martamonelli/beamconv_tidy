import ducc0

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import healpy as hp
from beamconv import ScanStrategy
import qpoint as qp

# reading cls from .txt file
cls = np.loadtxt('../ancillary/wmap7_r0p03_lensed_uK_ext.txt',
    unpack=True) # Cl in uK^2
ell, cls = cls[0], cls[1:]

# calculating alm from cls
lmax=128
np.random.seed(25) # why do I need it?
alm = hp.synalm(cls, lmax=lmax, new=True, verbose=True) # uK

# calculating map0 from alm
map0 = hp.alm2map(alm, 512)
hp.mollview(map0[0])
plt.savefig('map0.png')
plt.close()

# setting up ScanStrategy and its options
fwhm = 40
beam_opts = dict(lmax=lmax,
                 fwhm=fwhm,       # Gaussian co-pol beam, so only specify FWHM (arcmin)
                 btype='Gaussian')

ctime0 = 1510000000
mlen = 1 * 24 * 60 * 60        # Mission length in seconds

mmax = 2
nside_spin = 128
preview_pointing = False
verbose = True

ss = ScanStrategy(duration=mlen,
        external_pointing=True,
        sample_rate=19.1, # sample rate in Hz
        location='space',
        ctime0=ctime0)

scan_opts = dict(
        q_bore_func=ss.litebird_scan,
        ctime_func=ss.litebird_ctime,
        use_litebird_scan=True,
        q_bore_kwargs=dict(),
        ctime_kwargs=dict(),
        max_spin=mmax,
        nside_spin=nside_spin,
        preview_pointing=preview_pointing,
        verbose=verbose,
        save_tod=True)

# create a square grid of Gaussian beams (how is f.o.v. related to the number of beams?)
ss.create_focal_plane(nrow=1, ncol=1, fov=1, **beam_opts)
#ss.create_focal_plane(nrow=3, ncol=3, fov=3, **beam_opts)

# calculate tods in ten chunks (they are 10, right?)
chunks = ss.partition_mission(0.1 * ss.mlen * ss.fsamp)

# allocate and assign parameters for mapmaking
ss.allocate_maps(nside=128)

ss.scan_instrument_mpi(alm, **scan_opts)

# solve for the maps
maps, cond, proj = ss.solve_for_map(return_proj = True)

# parameters for generating noise with ducc0
sigma = 10
f_min=1e-4
f_knee=1e-1
f_samp=19.1
slope=-1.7
nsamp = int(0.1 * ss.mlen * ss.fsamp)

# setting up and fillind TODs (clean, noisy and noise-only) 
clean_TOD = np.empty((1,1,int(ss.mlen*ss.fsamp)))
noisy_TOD = np.empty((1,1,int(ss.mlen*ss.fsamp)))
noise_TOD = np.empty((1,1,int(ss.mlen*ss.fsamp)))

for dx in np.arange(1):
    for dy in np.arange(1):
        gen = ducc0.misc.OofaNoise(sigma, f_min, f_knee, f_samp, slope)
        for chunk in np.arange(10):
            inp = np.random.normal(0.,1.,(nsamp,))
            noise = gen.filterGaussian(inp)
            clean_data = ss.data(chunks[chunk],ss.beams[dx][dy],data_type='tod')
            noisy_data = clean_data + noise
            clean_TOD[dx,dy,int(chunk*ss.mlen*ss.fsamp/10):int((chunk+1)*ss.mlen*ss.fsamp/10)] = clean_data
            noisy_TOD[dx,dy,int(chunk*ss.mlen*ss.fsamp/10):int((chunk+1)*ss.mlen*ss.fsamp/10)] = noisy_data
            noise_TOD[dx,dy,int(chunk*ss.mlen*ss.fsamp/10):int((chunk+1)*ss.mlen*ss.fsamp/10)] = noise

# evaluating power spectrum
ps = np.abs(np.fft.fft(noise_TOD[0,0,:]))**2 / (nsamp*10)
time_step = 1. / f_samp
freqs = np.fft.fftfreq(noise_TOD[0,0,:].size, time_step)
ps_theory = sigma**2 * ((freqs**2+f_knee**2)/(freqs**2+f_min**2))**(-slope/2)

# plotting
#plt.plot(freqs[idx], ps[idx])
plt.loglog(freqs[:ps.size//2],ps[:ps.size//2])
plt.loglog(freqs[:ps.size//2],ps_theory[:ps.size//2])
plt.show()

plt.plot(noisy_TOD[0,0,:])
plt.plot(clean_TOD[0,0,:])
plt.show()

quit()

####################################################
# other stuff
####################################################
cond[cond == np.inf] = hp.UNSEEN
cart_opts = dict(unit=r'[$\mu K_{\mathrm{CMB}}$]')#, lonra=[-60, 40], latra=[-70, -40])
plt.figure(1)
hp.mollview(cond, min=2, max=5, **cart_opts)
plt.savefig('cond.png')
plt.close()

hp.mollview(proj[0])
plt.savefig('proj0.png')
plt.show()
plt.close()

print(np.min(maps[0].flatten()))
print(np.max(maps[0].flatten()))

hp.mollview(maps[0], min=-400, max=400, **cart_opts)
plt.savefig('map_I.png')
plt.close()

hp.mollview(maps[1], min=-10, max=10, **cart_opts)
plt.savefig('map_Q.png')
plt.close()
