
# use this to get full detector images for flat field
import sys
import os

import numpy as np
import h5py
import argparse
from scipy.signal import argrelmax
import pylab as plt

import psana

def flatten_and_mask_shots(shot, mask):
    
    size = shot.shape[0]*shot.shape[1]
    flat_mask = mask.reshape(size)
    flat_shot =  shot.reshape(size)*flat_mask
    
    return flat_shot[flat_mask]

parser = argparse.ArgumentParser(description='Data grab: get radial Radial Profiles.\n\
    Use them to decide peak range for hit detection.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-r', '--run', type=int, required=True, help='run number to process')
parser.add_argument('-m', '--maxcount', type=int, default=0, help='max shots to process')
parser.add_argument('-s', '--start', type=int, default=0, help='first event to process')
parser.add_argument('-f', '--fname', type=str, required=True, help='output basename')
parser.add_argument('-d', '--out_dir', type=str, required=True,
help='dir in which to store the polar data output')

parser.add_argument('-i', '--intensity_treshold', default=0.01, type=float,
help='intensity_treshold for rejecting shots')

args = parser.parse_args()
    
#############parameters to change
# img_sh: n-by-m pixel shape of the cspad
# cent_fname: .npy file storing the center of the detector in pixel unit, use NP rings to find center
# ds_str: change to the correct experiment number

# run number passed as string
run = args.run
start = args.start

#~~~~~analysis  parameters BEGIN

#~~~ data get parameters
#load the data events for the given run
##########################################
#Need to change the experiment number to cxilr67##
##########################################
ds_str = 'exp=cxilp6715:run=%d:smd' % run
ds = psana.MPIDataSource(ds_str)
events = ds.events()

# open the detector obj
# change cspad alias!
#############################################
cspad = psana.Detector('CxiDs1.0:Cspad.0')
#############################################

# make some output dir
outdir = args.out_dir
if not os.path.exists(outdir):
    os.makedirs(outdir)

# make output file
prefix = args.fname
out_fname = os.path.join( outdir, prefix)

#small dat
smldata = ds.small_data(out_fname)

# load mask
mask = np.load('/reg/d/psdm/cxi/cxilp6715/results/shared_files/mask_rough4.npy')

count = 0
seen_evts = 0

for i,evt in enumerate(events):
#   keep this first, i should never be < 0
    if i < start: 
        #print ("skipping event %d/%d"%(i+1, start))
        continue

    img = cspad.image(evt)
    
    seen_evts += 1
    if seen_evts == args.maxcount:
        break
    

    if img is None:
        print("img is None")
        continue
    #flatten
    flat_shot = flatten_and_mask_shots(img,mask)
    ave_int = flat_shot.mean()

    if ave_int<args.intensity_treshold:
        print ("intensity is too low, skip")
        continue

#   save
    smldata.event(flat_img = flat_shot)
    smldata.event(ave_tot_int = ave_int)

    count+=1
    print("Images processed: %d out of %d events..."%(count,i+1))

smldata.save()


