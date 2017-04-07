from astropy.table import Table,hstack
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import h5py

MISTFILE_default = '/Users/pcargile/Astro/MIST/MIST_v1.0/MIST_full.h5'
# MISTFILE_default =  '/n/regal/conroy_lab/pac/MISTFILES/MIST_full_1.h5'

def DM(distance):
	#distance in parsecs
	return 5.0*np.log10(distance)-5.0

class Redden(object):
	def __init__(self,stripeindex=None):
 		if stripeindex == None:
 			BCfile = MISTFILE_default
 		else:
 			BCfile = '/n/regal/conroy_lab/pac/MISTFILES/MIST_full_{0}.h5'.format(stripeindex)

 		# read in MIST hdf5 table
 		MISTh5 = h5py.File(BCfile,'r')
 		# determine the BC datasets
 		BCTableList = [x for x in MISTh5.keys() if x[:3] == 'BC_']
 		# read in each BC dataset and pull the photometric information
 		for BCT in BCTableList:
	 		BCTABLE = Table(np.array(MISTh5[BCT]))
			if BCT == BCTableList[0]:
				BC = BCTABLE.copy()
			else:
				BCTABLE.remove_columns(['Teff', 'logg', '[Fe/H]', 'Av', 'Rv'])
				BC = hstack([BC,BCTABLE])

 		BC_AV0 = BC[BC['Av'] == 0.0]

		self.bands = BC.keys()
		[self.bands.remove(x) for x in ['Teff', 'logg', '[Fe/H]', 'Av', 'Rv']]

		self.redintr = LinearNDInterpolator(
			(BC['Teff'],BC['logg'],BC['[Fe/H]'],BC['Av']),
			np.stack([BC[bb] for bb in self.bands],axis=1),
			rescale=True
			)
		self.redintr_0 = LinearNDInterpolator(
			(BC_AV0['Teff'],BC_AV0['logg'],BC_AV0['[Fe/H]']),
			np.stack([BC_AV0[bb] for bb in self.bands],axis=1),
			rescale=True
			)

	def red(self,Teff=5770.0,logg=4.44,FeH=0.0,band='V',Av=0.0):

		if (Teff > 500000.0):
			Teff = 500000.0
		if (Teff < 2500.0):
			Teff = 2500.0

		if (logg < -4.0):
			logg = -4.0
		if (logg > 9.5):
			logg = 9.5

		if (FeH > 0.5):
			FeH = 0.5
		if (FeH < -2.0):
			FeH = -2.0

		inter0 = self.redintr_0(Teff,logg,FeH)
		interi = self.redintr(Teff,logg,FeH,Av)

		bandsel = np.array([True if x == band else False for x in self.bands],dtype=bool)
		vsel = np.array([True if x == 'Bessell_V' else False for x in self.bands],dtype=bool)
		A_V = interi[vsel]-inter0[vsel]
		A_X = interi[bandsel]-inter0[bandsel]
		return (A_X/A_V)*Av
