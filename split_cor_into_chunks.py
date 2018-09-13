
import numpy as np
import h5py
import os

import argparse
import sys
import glob

from loki.RingData.DiffCorr import DiffCorr

parser = argparse.ArgumentParser(description='cluster shots by kmeans')
parser.add_argument('-s', '--sample', type=str, required=True, help='run number to process')
parser.add_argument('-d', '--data_dir', type=str,
    default = '/reg/d/psdm/cxi/cxilp6715/results/denoise_polar_intensity/diagnostics/all_pca_maskByRun/',
help='dir in which the pca denoised data are')
parser.add_argument('-o', '--out_dir', type=str,required=True,
help='dir in which to store the output')
parser.add_argument('-m', '--mask_dir', type=str,
    default='/reg/d/psdm/cxi/cxilp6715/results/combined_tables/used_in_OE/',
help='dir in which the binned masks are, i.e. the polar image directory')
parser.add_argument('-n', '--num_chunks', type=int,required=True,
help='number of chunks to aplit the data into')

args = parser.parse_args()

num_chunks = args.num_chunks

sample = args.sample
data_dir = args.data_dir
save_dir =args.out_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print "saving in %s..."%save_dir

all_files =glob.glob('%s/*PCA*'%os.path.join(data_dir,sample))
print all_files
run_nums = [int(os.path.basename(ff).split('_')[0].split('n')[-1]) for ff in all_files]

outfname = os.path.join(save_dir,'%s_chunks_split_maskDivide.h5'%sample)

# if os.path.isfile(outfname):
#     print ('will not overwrite file')
#     sys.exit()

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

        all_ave_cors=[]

        all_nums=[]

    

        for qidx in range(35):
            print('run%d q%d'%(run,qidx))
            if 'num_pca_cutoff2' in f['q%d'%qidx].keys():
                pca_num = f['q%d'%qidx]['num_pca_cutoff2'].value
            else:
                pca_num = f['q%d'%qidx]['num_pca_cutoff'].value
            #take half of everything
            num = f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'].shape[0]

            chunk_size = int(num/num_chunks)
            chunk_cor = []
            chunk_num = []

            for ic in range(num_chunks):
                cc = f['q%d'%qidx]['pca%d'%pca_num]['all_train_difcors'][:(ic+1)*chunk_size,0]/mask_cor[qidx][None,:]
                ave = cc.mean(0)
                chunk_cor.append(ave)
                chunk_num.append((ic+1)*chunk_size)

            all_ave_cors.append(np.array(chunk_cor))
            all_nums.append(np.array(chunk_num))


        f_out.create_dataset('run%d/ave_cor_byChunk'%run,data=np.array(all_ave_cors))
        f_out.create_dataset('run%d/num_shots'%run,data=np.array(all_nums))

        

        print np.array(all_ave_cors).shape

    # aggregate all the averages
    print('aggregating results')
    ave_cor1 =[]
    total_shots1=[]

    keys=[kk for kk in f_out.keys() if kk.startswith('run')]

    for kk in keys:
        ave_cor1.append(f_out[kk]['ave_cor_byChunk'].value)
        total_shots1.append(f_out[kk]['num_shots'].value)
    
    ave_cor1=np.array(ave_cor1)
    total_shots1=np.array(total_shots1)
    
    chunk_ave_cors =[]
    chunk_nShots=[]
    for ic in range(num_chunks):
        shots = ave_cor1[:,:,ic,:]
        nn = total_shots1[:,:,ic]

        cor1 = (shots *(nn.astype(float)/nn.sum(0)[None,:])[:,:,None]).sum(0)

        chunk_ave_cors.append(cor1)
        chunk_nShots.append(nn)
    chunk_ave_cors = np.array(chunk_ave_cors)
    chunk_nShots = np.array(chunk_nShots)

    f_out.create_dataset('running_ave_cor',data=chunk_ave_cors)
    f_out.create_dataset('running_num_shots',data=chunk_nShots)

    print('Done!')