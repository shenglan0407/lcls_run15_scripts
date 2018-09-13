
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

import glob


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
# q index
qmin = args.qmin
qmax = args.qmax

if qmax is None:
    qcluster_inds = [qmin]
else:
    qcluster_inds = range(qmin,qmax+1) # qmax is included


# output file to save data
outfiles = glob.glob(os.path.join(save_dir,'run%d_PCA-denoise_*.h5'%run_num))
n_files = len(outfiles)
print n_files
for jj in range(n_files):
    out_file =os.path.join(save_dir,'run%d_PCA-denoise_%d.h5'%(run_num,jj) )
    f_out = h5py.File(out_file,'r')
    try:
        nn=f_out['q10']['num_pca_cutoff'].value
    except KeyError:
        print("skipping run %d"%run_num)
        sys.exit()
    out_file2 = run_file.replace('.tbl','_standard_errors_%d.h5'%jj)
    f_out2 = h5py.File(os.path.join(save_dir, out_file2),'w')




    for qidx in qcluster_inds:
        print('computing standard errors for qidx %d'%qidx)
        q_group = 'q%d'%qidx
        if q_group not in f_out.keys():
            f_out.create_group(q_group)
        
        this_mask = mask[qidx][None,:]
        print('normaling shots...')
        
        #### denoise at critical component
        grp=f_out['q%d'%qidx]
        if 'num_pca_cutoff2' in grp.keys():
            nn=grp['num_pca_cutoff2'].value
        else:
            nn=grp['num_pca_cutoff'].value
            print("pca critical component: %d"%nn)

        
        intershot_difcor= grp['pca%d'%nn]['all_train_difcors'].value/mask_cor[qidx][None,None,:]

        a = range(intershot_difcor.shape[0])
        np.random.shuffle(a)
        x1 = a[:len(a)/2]
        x2 = a[len(a)/2:]

        e1=intershot_difcor[x1].std(0)/np.sqrt(len(x1))
        e2=intershot_difcor[x2].std(0)/np.sqrt(len(x2))
        f_out2.create_dataset('q%d/pca%d/random_split_err1'%(qidx,nn)
            ,data=e1)
        f_out2.create_dataset('q%d/pca%d/random_split_err2'%(qidx,nn)
            ,data=e2)
        f_out2.create_dataset('q%d/pca%d/standard_err'%(qidx,nn)
            ,data=intershot_difcor.std(0)/np.sqrt(intershot_difcor.shape[0]))


        del intershot_difcor
      

    f_out.close()
    f_out2.close()

print ("done!")
