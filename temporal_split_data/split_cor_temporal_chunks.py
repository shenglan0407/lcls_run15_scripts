import numpy as np
import h5py
import os

import argparse
import sys
import glob

from loki.RingData.DiffCorr import DiffCorr

parser = argparse.ArgumentParser(description='cluster shots by kmeans')
parser.add_argument('-s', '--sample', type=str, required=True, help='run number to process')
parser.add_argument('-d', '--data_dir', type=str,required=True,
help='dir in which the pca denoised data are')
parser.add_argument('-o', '--out_dir', type=str,required=True,
help='dir in which to store the output')
parser.add_argument('-m', '--mask_dir', type=str,required=True,
help='dir in which the binned masks are, i.e. the polar image directory')
parser.add_argument('-r', '--runs', type=str,required=True,
help='file in which run numbers are stored')
parser.add_argument('-n', '--n_shots', type=int,required=True,
help='how many shots to grab per file')

args = parser.parse_args()


sample = args.sample
data_dir = args.data_dir
save_dir =args.out_dir

run_nums = np.load(args.runs)
n_shots = int(args.n_shots)
print n_shots

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print "saving in %s..."%save_dir
# 
outfname = os.path.join(save_dir,'%s_temporal_chunk.h5'%sample)


with h5py.File(outfname,'a') as f_out:
    for run in run_nums:

        print("splitting shots for run %d"%run)
        f = h5py.File('%s/run%d_PCA-denoise.h5'%(os.path.join(data_dir,sample),run),'r')
        try:
            pca_num = f['q10']['num_pca_cutoff'].value
        except KeyError:
            print("skipping run %d"%run)
            continue
        if 'run%d'%run in f_out.keys():
            print("already seen this run, skip!")
            continue
        
        f_out.create_group('run%d'%run)
        ##### load the mask used for this run
        f_mask = h5py.File(os.path.join(args.mask_dir,'run%d.tbl'%run),'r')

        mask = f_mask['polar_mask_binned'].value
        mask = (mask==mask.max())
        mask.shape
        # do the mask cor
        qs = np.linspace(0,1,mask.shape[0])
        dc=DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
        mask_cor = dc.autocorr().mean(0)
        
        f_mask.close()


        all_ave_cors1=[]
        all_nums1=[]
        all_err1=[]


        for qidx in range(35):
            print('run%d q%d'%(run,qidx))
            if 'num_pca_cutoff2' in f['q%d'%qidx].keys():
                pca_num = f['q%d'%qidx]['num_pca_cutoff2'].value
            else:
                pca_num = f['q%d'%qidx]['num_pca_cutoff'].value

            cc = f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'][:n_shots,0]/mask_cor[qidx][None,:]
            ave1 = cc.mean(0)
            std1 = cc.std(0)
            n1 = cc.shape[0]
            all_nums1.append(n1)
            all_ave_cors1.append(ave1)
            all_err1.append(std1/np.sqrt(n1))


        f_out.create_dataset('run%d/ave_cor1'%run,data=np.array(all_ave_cors1))
        f_out.create_dataset('run%d/num_shots'%run,data=np.array(all_nums1))
        f_out.create_dataset('run%d/err1'%run,data=np.array(all_err1))
        print np.array(all_err1).shape

print('Done!')