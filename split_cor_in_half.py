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

args = parser.parse_args()


sample = args.sample
data_dir = args.data_dir
save_dir =args.out_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print "saving in %s..."%save_dir

all_files =glob.glob('%s/*PCA*'%os.path.join(data_dir,sample))
print all_files
run_nums = [int(os.path.basename(ff).split('_')[0].split('n')[-1]) for ff in all_files]

outfname = os.path.join(save_dir,'%s_midRun_split_maskDivide.h5'%sample)

if os.path.isfile(outfname):
    print ('will not overwrite file')
    sys.exit()

with h5py.File(outfname,'a') as f_out:
    for run in run_nums:
        print("splitting shots for run %d"%run)
        f = h5py.File('%s/run%d_PCA-denoise.h5'%(os.path.join(data_dir,sample),run),'r')
        try:
            pca_num = f['q10']['num_pca_cutoff'].value
        except KeyError:
            print("skipping run %d"%run)
            continue

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

        f_out.create_group('run%d'%run)

        all_ave_cors1=[]
        all_ave_cors2=[]

        all_nums1=[]
        all_nums2=[]

        all_err1=[]
        all_err2=[]

        for qidx in range(35):
            print('run%d q%d'%(run,qidx))
            
            pca_num = f['q%d'%qidx]['num_pca_cutoff'].value
            #take half of everything
            num = f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'].shape[0]

            cc = f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'][:num/2,0]/mask_cor[qidx][None,:]
            ave1 = cc.mean(0)
            std1 = cc.std(0)
            n1 = cc.shape[0]
            all_nums1.append(n1)
            all_ave_cors1.append(ave1)
            all_err1.append(std1/np.sqrt(n1))

            cc = f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'][num/2:,0]/mask_cor[qidx][None,:]
            ave2 = cc.mean(0)
            std2 = cc.std(0)
            n2 = cc.shape[0]
            all_nums2.append(n2)
            all_ave_cors2.append(ave2)
            all_err2.append(std2/np.sqrt(n2))

        f_out.create_dataset('run%d/ave_cor1'%run,data=np.array(all_ave_cors1))
        f_out.create_dataset('run%d/ave_cor2'%run,data=np.array(all_ave_cors2))
        f_out.create_dataset('run%d/num_shots1'%run,data=np.array(all_nums1))
        f_out.create_dataset('run%d/num_shots2'%run,data=np.array(all_nums2))
        f_out.create_dataset('run%d/err1'%run,data=np.array(all_err1))
        f_out.create_dataset('run%d/err2'%run,data=np.array(all_err2))
        print np.array(all_err1).shape

    # aggregate all the averages
    print('aggregating results')
    ave_cor1 =[]
    ave_cor2=[]
    total_shots1=[]
    total_shots2 = []
    err1 = []
    err2 = []

    keys=f_out.keys()

    for kk in keys:
        ave_cor1.append(f_out[kk]['ave_cor1'].value)
        ave_cor2.append(f_out[kk]['ave_cor2'].value)

        total_shots1.append(f_out[kk]['num_shots1'].value)
        total_shots2.append(f_out[kk]['num_shots2'].value)

        err1.append(f_out[kk]['err1'].value)
        err2.append(f_out[kk]['err2'].value)
    ave_cor1=np.array(ave_cor1)
    ave_cor2=np.array(ave_cor2)

    err1 = np.array(err1)
    err2 = np.array(err2)

    total_shots1=np.array(total_shots1)
    total_shots2=np.array(total_shots2)

    cor1 = (ave_cor1 *(total_shots1.astype(float)/total_shots1.sum(0)[None,:])[:,:,None]).sum(0)
    cor2 = (ave_cor2 *(total_shots2.astype(float)/total_shots2.sum(0)[None,:])[:,:,None]).sum(0)

    total_err1 = np.sqrt((err1**2 *total_shots1[:,:,None]**2).sum(0))/total_shots1.sum(0)[:,None]
    total_err2 = np.sqrt((err2**2 *total_shots2[:,:,None]**2).sum(0))/total_shots2.sum(0)[:,None]

    f_out.create_dataset('ave_cor1',data=cor1)
    f_out.create_dataset('ave_cor2',data=cor2)
    f_out.create_dataset('err1',data=total_err1)
    f_out.create_dataset('err2',data=total_err2)

    f_out.create_dataset('num_shots1',data=total_shots1)
    f_out.create_dataset('num_shots2',data=total_shots2)

    print('Done!')