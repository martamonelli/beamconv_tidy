import numpy as np 
from matplotlib import pyplot as plt

bandcenter = 140.0
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

#Plot Mueller
fig, ax = plt.subplots(4,4, sharex=True, figsize=(10,8))
stokes_string = ["I","Q","U","V"]

ax[0,0].axhline(y=1,color='k',linestyle='dashed')
ax[0,1].axhline(y=0,color='k',linestyle='dashed')
ax[0,2].axhline(y=0,color='k',linestyle='dashed')
ax[0,3].axhline(y=0,color='k',linestyle='dashed')
ax[1,0].axhline(y=0,color='k',linestyle='dashed')
ax[1,1].axhline(y=1,color='k',linestyle='dashed')
ax[1,2].axhline(y=0,color='k',linestyle='dashed')
ax[1,3].axhline(y=0,color='k',linestyle='dashed')
ax[2,0].axhline(y=0,color='k',linestyle='dashed')
ax[2,1].axhline(y=0,color='k',linestyle='dashed')
ax[2,2].axhline(y=-1,color='k',linestyle='dashed')
ax[2,3].axhline(y=0,color='k',linestyle='dashed')
ax[3,0].axhline(y=0,color='k',linestyle='dashed')
ax[3,1].axhline(y=0,color='k',linestyle='dashed')
ax[3,2].axhline(y=0,color='k',linestyle='dashed')
ax[3,3].axhline(y=-1,color='k',linestyle='dashed')

for i in range(4):
    for j in range(4):
        ax[i,j].plot(trans_rec[freqs_array,0], mueller[freqs_array,i,j], color='teal')
        ax[i,j].set_title(stokes_string[i]+stokes_string[j])
        
fig.tight_layout()
plt.show()
