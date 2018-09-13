# cluster and save norm_shots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import h5py
import os

from loki.utils.postproc_helper import *
from loki.RingData import DiffCorr
from loki.utils import stable

import argparse
import sys
sys.path.append('/reg/neh/home/qiaoshen/GitHub/lcls_run16_analysis_scripts/flatfield_calibrations')
from cali_utils import *

parser = argparse.ArgumentParser(description='cluster shots by kmeans')
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-d', '--data_dir', type=str,required=True,
help='dir in which the polar data are')
parser.add_argument('-o', '--out_dir', type=str,required=True,
help='dir in which to store the output')
parser.add_argument('-p', '--num_pca', type=int,default=10,
    help='number of pca n_components to keep for kmeans clustering')
parser.add_argument('-k', '--num_kmeans', type=int,default=15,
    help='number of kmeans clusters')

parser.add_argument('-q', '--qmin', type=int,required=True,
    help='min index of q')
parser.add_argument('-m', '--qmax', type=int,required=True,
    help='max index of q')

args = parser.parse_args()


def normalize(d):
    x=d.copy()
    x-=x.min()
    return x/(x.max()-x.min())
#range of q to cluster for
qmin = args.qmin
qmax = args.qmax+1

num_pca = args.num_pca
num_kmeans = args.num_kmeans
# run number
run_num = args.run
# output name
PI_dir = args.data_dir

save_dir = os.path.join( args.out_dir, "pca%d_kmeans%d"%(num_pca, num_kmeans))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print save_dir

fname_output = os.path.join(save_dir,'run%d_clustered_shots.h5'%run_num)
# if os.path.exists(fname_output):
#     print(fname_output)
#     print('file already exist for this run. will not overwrite. quit!')
#     sys.exit()

f_out = h5py.File(fname_output,'a')

# load q values ###################### need to change this!!!!!
# qvalue = np.load('/reg/neh/home/qiaoshen/dissertation_notebooks/qvalues.npy')
#dummy values
qvalue = np.linspace(0.1,1,len(range(qmin,qmax)))
# ###### load shots from a run
f_PI=h5py.File(os.path.join(PI_dir,'run%d.tbl'%run_num),'r')

mask = f_PI['polar_mask_binned'].value
mask = (mask==mask.max())[range(qmin,qmax)]
shots = f_PI['polar_imgs'].value[:, qmin:qmax]



########################################################
#normaliza shots while masking bright spots

outlier_threshold = 2.5
norm_shots = norm_all_shots(shots.astype(np.float64),mask, outlier_threshold)
# filter out abnormaly large norm_shots
x = norm_shots.mean(-1)
s = x.mean(0)
m = x.std(0)
select = None
for ii in range(x.shape[1]):
    if select is None:
        select = np.abs(x[:,ii]-s[ii])<m[ii]*10
    select *= (np.abs(x[:,ii]-s[ii])<m[ii]*10)
select = select.astype(bool)
print select.shape, norm_shots.shape
print select.sum()/float(norm_shots.shape[0])
# f_out.close()

if norm_shots.shape[0]%2>0:
    norm_shots=norm_shots[:-1]
######## load mask and normalize the shots
phi_offset=30
num_phi=norm_shots.shape[-1]
qs = np.linspace(0,1,shots.shape[1])
dc=DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
mask_corr=dc.autocorr()
##### compute single-shot correlations
dc = DiffCorr(norm_shots-norm_shots.mean(0)[None,:,:],
  qs,0,pre_dif=True)
corr = dc.autocorr()
print corr.shape

corr/=mask_corr
corr=corr[:,:,phi_offset:num_phi/2-phi_offset]

diff_shots = norm_shots[::2]-norm_shots[1::2]
dc=DiffCorr(diff_shots,qs,0,pre_dif=True)
no_cluster_ac= (dc.autocorr()/mask_corr).mean(0)
f_out.create_dataset('raw_corrs',data=no_cluster_ac)

####### do PCA on the shots and cluster them with Kmeans
for qidx in range(qmin,qmax):
    print('cluster for qidx %d'%qidx)
    f_out.create_group('q%d'%qidx)
    ####### do PCA on the shots and cluster them with Kmeans
    pca=PCA(n_components=num_pca)

    new_corr=pca.fit_transform(corr[:,qidx-qmin,:])
    kmeans=KMeans(n_clusters=args.num_kmeans)
    kmeans.fit(new_corr)

    # sort the polar intensities into cluster
    # compute cluster correlations
    all_ac=[]
    num_shots=[]

    ave_cluster_corr = []
    for ll in sorted(np.unique(kmeans.labels_)):


        ss=norm_shots[kmeans.labels_==ll, qidx-qmin]

        if ss.shape[0]<2:
            continue
        if ss.shape[0]%2>0:
            ss = ss[:-1]

        f_out.create_dataset('q%d/norm_shots_%d'%(qidx,ll),data=ss)

        this_mask=mask[qidx-qmin][None,:]
        # mask correlations
        # dummy qs
        qs = np.array([0.1])
        dc=DiffCorr(this_mask[None,:,:],qs,0,pre_dif=True)
        mask_ac=dc.autocorr()
        
        # difference correlations of the cluster
        ss_diff = ss[::2]-ss[1::2]
        print ss_diff.shape, qs.shape
        dc=DiffCorr(ss_diff[:,None,:],qs,0,pre_dif=True)
        ac=dc.autocorr()/mask_ac
        
        all_ac.append(ac[:,0])
        num_shots.append(ss.shape[0])
        
        ave_cluster_corr.append(ac.mean(0)[0])
        
    combined_ac=np.concatenate(all_ac).mean(0)


    # compute and store asymmetries
    cluster_cor_asym =[] 
    for ii ,cc in enumerate(ave_cluster_corr):
        nc=normalize(cc[phi_offset:num_phi/2-phi_offset])
        cluster_cor_asym.append( (np.abs(nc-nc[::-1])).mean() )
    cluster_cor_asym=np.array(cluster_cor_asym)
    ave_cluster_corr=np.array(ave_cluster_corr)

    f_out.create_dataset('q%d/asym'%qidx, data=cluster_cor_asym)
    f_out.create_dataset('q%d/ave_clus_cor'%qidx, data=ave_cluster_corr)
    f_out.create_dataset('q%d/clus_nshots'%qidx,data = np.array(num_shots) )
    f_out.create_dataset('q%d/ave_cor'%qidx,data = combined_ac )

f_out.close()
print('Done!')

