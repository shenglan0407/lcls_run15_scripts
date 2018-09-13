
# take a run, do PCA at each q

# save the shots into train and test, save the 
import h5py
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import argparse
import numpy as np

import sys

from loki.RingData.DiffCorr import DiffCorr


parser = argparse.ArgumentParser(description='Compute difference correlation by pairing single intensity correlations.')
parser.add_argument('-r','--run', type=int,
                   help='run number')
parser.add_argument('-t','--samp_type', type=int,
                   help='type of data/n \
# Sample IDs\n\
# -1: Silver Behenate smaller angle\n\
# -2: Silver Behenate wider angle\n\
# 0: GDP buffer\n\
# 1: ALF BUffer\n\
# 2: GDP protein\n\
# 3: ALF protein\n\
# 4: Water \n\
# 5: Helium\n\
# 6: 3-to-1 Recovered GDP')

parser.add_argument('-q','--qmin', type=int,
                   help='index of minimum q used for pairing or the only q used for pairing')

parser.add_argument('-u','--qmax', type=int, default=None,
                   help='index of max q used for pairing or None')

parser.add_argument('-o','--out_dir', type=str,required=True,
                   help='output dir to save in, overwrites the sample type dir')

parser.add_argument('-d','--data_dir', type=str, default = '/reg/d/psdm/cxi/cxilp6715/results/combined_tables/finer_q',
                   help='where to look for the polar data')

parser.add_argument('-n', '--num_chunks', type=int,required=True,
help='number of chunks to aplit the data into')


def sample_type(x):
    return {-1:'AgB_sml',
    -2:'AgB_wid',
     0:'GDP_buf',
     1:'ALF_buf',
     2:'GDP_pro',
     3:'ALF_pro',
     4:'h2o',
     5:'he',
     6:'3to1_rec_GDP_pro'}[x]

def normalize_shot(ss, this_mask):
    if ss.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        ss = ss.astype(np.float64)
    
    ss *=this_mask
    mean_ss = ss.sum(-1)/this_mask.sum(-1) 
    ss = ss-mean_ss[:,None]
    return np.nan_to_num(ss*this_mask)

def reshape_unmasked_values_to_shots(shots,mask):
    # this takes vectors of unmasked values, and reshaped this into their masked forms
    # mask is 2D, shots are 1D
    assert(shots.shape[-1]==np.sum(mask) )
    flat_mask = mask.flatten()
    reshaped_shots = np.zeros( (shots.shape[0],mask.size), dtype=shots.dtype)
    
    reshaped_shots[:, flat_mask==1] = shots
    
    return reshaped_shots.reshape((shots.shape[0],mask.shape[0],mask.shape[1]))
    


args = parser.parse_args()



run_num = args.run
num_chunks = args.num_chunks


if args.samp_type not in [-1,-2,0,1,2,3,4,5,6]:
    print("Error!!!! type of sample does not exist")
    sys.exit()
else:
    sample = sample_type(args.samp_type)
# import run file

data_dir = args.data_dir

save_dir = os.path.join( args.out_dir, sample)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print save_dir


run_file = "run%d.tbl"%run_num

# load the run
f = h5py.File(os.path.join(data_dir, run_file), 'r')

# output file to save data
out_file = run_file.replace('.tbl','_PCA-denoise.h5')
f_out = h5py.File(os.path.join(save_dir, out_file),'r')
try:
    nn=f_out['q0']['num_pca_cutoff'].value
except KeyError:
    print("skipping run %d"%run_num)
    sys.exit()
out_file2 = run_file.replace('.tbl','_chunks_intershot_uncertainty.h5')
f_out2 = h5py.File(os.path.join(save_dir, out_file2),'w')

if 'polar_mask_binned' in f.keys():
    mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)
else:
    print("there is no mask stored with the shots")
    sys.exit()
    # mask = np.load('/reg/d/psdm/cxi/cxilp6715/results/shared_files/binned_pmask_basic.npy')
# do the mask cor
qs = np.linspace(0,1,mask.shape[0])
dc=DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
mask_cor = dc.autocorr().mean(0)

PI = f['polar_imgs']
# filter by photon energy. If the photon energy of the shot if not within 100 EV of the average, do not use
photon_energy=np.nan_to_num(f['ebeam']['photon_energy'].value)
mean_E=photon_energy.mean()
E_sigma=100.
shot_tage_to_keep=np.where( (photon_energy> (mean_E-E_sigma))\
    +(photon_energy< (mean_E-E_sigma)) )[0]
print('Num of shots to be used: %d'%(shot_tage_to_keep.size))

# figure which qs are used for pairing
qmin = args.qmin
qmax = args.qmax

if qmax is None:
    qcluster_inds = [qmin]
else:
    qcluster_inds = range(qmin,qmax+1) # qmax is included

# this script on does the clustering only



# normalize all the shots at each q index

for qidx in qcluster_inds:
    print('PCA denoising for qidx %d'%qidx)
    q_group = 'q%d'%qidx
    if q_group not in f_out.keys():
        f_out.create_group(q_group)
    shots=PI[:,qidx,:][shot_tage_to_keep,None,:]
    this_mask = mask[qidx][None,:]
    print('normaling shots...')
    norm_shots = np.zeros_like(shots)
    for idx,ss in enumerate(shots):
        norm_shots[idx]=normalize_shot(ss,this_mask)
    # do we want to normalize by the entire range of intensity?
    # divide into Train and test
    num_shots = norm_shots.shape[0]
    cutoff = int(num_shots*0.1) # use 10% of the shots as testing set
    partial_mask = this_mask.copy()
    Train = norm_shots[cutoff:, partial_mask==1]
   
    print ("%d train shots"%(Train.shape[0]))


    qvalues = np.linspace(0,1,partial_mask.shape[0])
  
    if 'pca_components' in f_out[q_group].keys():
        #load the components previously save the then do the transformations
        components = f_out['q%d/pca_components'%qidx].value
        _m = Train.astype(np.float64).mean(0)
        new_Train = (Train.astype(np.float64)-_m).dot(components.T)

    else:
        print('no PCA components, quit!')
        sys.exit()
    # get back the masked images and components
 
    masked_mean_train =reshape_unmasked_values_to_shots(Train,partial_mask).mean(0)


    #### denoise at critical component
    grp=f_out['q%d'%qidx]
    if 'num_pca_cutoff2' in grp.keys():
        nn=grp['num_pca_cutoff2'].value

    else:
        nn=grp['num_pca_cutoff'].value

    Train_noise = new_Train[:,:nn].dot(components[:nn])
    denoise_Train= reshape_unmasked_values_to_shots(Train-Train_noise-Train.mean(0)[None,:]
                                                , partial_mask)


    # create inter-shots
    f_out2.create_group(q_group)
    print("estimating errors using intershots...")
    
    #####chunk it up
    num = Train_noise.shape[0]
    chunk_size = int(num/num_chunks)
    chunk_num = []
    chunk_err=[]
    for ic in range(num_chunks):
        dT = denoise_Train[:(ic+1)*chunk_size]
        if dT.shape[0]%2>0:
            dT=dT[1:]
        inter_shots = dT[::2]-dT[1::2]

        if inter_shots.shape[0]%2>0:
            inter_shots=inter_shots[1:]
        inter_shots = inter_shots[::2]-inter_shots[1::2]

        dc=DiffCorr(inter_shots,qvalues,0,pre_dif=True)
        intershot_difcor= dc.autocorr()/mask_cor[qidx][None,None,:]

        e1=intershot_difcor.std(0)/np.sqrt(intershot_difcor.shape[0])
        chunk_err.append(e1)
        chunk_num.append(intershot_difcor.shape[0])

        del inter_shots
        del intershot_difcor

    f_out2.create_dataset('q%d/pca%d/err_byChunk'%(qidx,nn)
        ,data=np.array(chunk_err) )
    f_out2.create_dataset('q%d/pca%d/num_shots'%(qidx,nn)
        ,data=np.array(chunk_num) )

   
    del shots
    del norm_shots

print ("done!")
f_out.close()
f_out2.close()