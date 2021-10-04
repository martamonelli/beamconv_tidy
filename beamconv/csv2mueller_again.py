# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

trans_rec = np.genfromtxt('MFT_HWP.csv', delimiter=';')

####################################################################
# I changed Alex's definitions!

jones = np.zeros((trans_rec.shape[0],2,2), dtype='complex128')
jones[:,0,0] = trans_rec[:,1]**0.5
jones[:,0,1] = 10**(trans_rec[:,6]/10.)
jones[:,1,0] = 10**(trans_rec[:,6]/10.)
jones[:,1,1] = -trans_rec[:,3]**0.5*np.exp(1j*np.radians(trans_rec[:,5]-180))
np.save("mft_hwp_jones_matrix.npy", jones)

#Plot Jones
plt.figure(1)
plt.plot(trans_rec[:,0], jones[:,0,0])
plt.plot(trans_rec[:,0], jones[:,0,1])
plt.plot(trans_rec[:,0], np.real(jones[:,1,1]))
plt.plot(trans_rec[:,0], np.imag(jones[:,1,1]), 'g:')
plt.xlabel("Frequency (GHz)")
plt.ylabel(r"$J_{ij}$")
plt.legend([r"$J_{11}$", r"$J_{12}$", r"$Re(J_{22})$", r"$Im(J_{22}$)"])
plt.title("MFT HWP")
plt.savefig('MFT_HWP_jones.png')
plt.close()

####################################################################
# From Jones to Mueller matrix elements

###From Tom Hileman's transfer_matrix.py
Sigma = []#Pauli + Indentity
Sigma.append( np.array(( (1,0),(0,1)), dtype=complex)) # identity matrix
Sigma.append( np.array(( (1,0),(0,-1)), dtype=complex))
Sigma.append( np.array(( (0,1),(1,0)), dtype=complex))
Sigma.append( np.array(( (0,-1j),(1j,0)), dtype=complex)) # Need to multiply by -1 to change back to normal.

mueller = np.zeros((trans_rec.shape[0],4,4))
            
for i in range(trans_rec.shape[0]):
    for j in range(4):
        for k in range(4):
            temp = .5*np.trace( np.dot(Sigma[j], np.dot(jones[i], np.dot(Sigma[k], jones[i].conj().transpose()))))
            mueller[i,j,k] = np.real(temp)

np.save("mft_hwp_mueller_matrix.npy", mueller)

#Plot Mueller
fig, ax = plt.subplots(4,4, sharex=True)
stokes_string = ["I","Q","U","V"]

for i in range(4):
    for j in range(4):
        ax[i,j].plot(trans_rec[:191-89,0], mueller[:191-89,i,j])
        ax[i,j].set_title(stokes_string[i]+stokes_string[j])
plt.savefig("MFT_HWP_mueller.png")
plt.close()
