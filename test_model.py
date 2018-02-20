#!/usr/bin/env python

import matplotlib
#matplotlib.use('Agg')
from keras.models import model_from_json
from keras import optimizers
from keras import callbacks
from keras import metrics
from scipy import misc
import time
import datetime
import os
import psutil
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import moviepy.editor as mp
import c3d_model_new2 as c3d_model
import sys
import keras.backend as K
import sklearn.preprocessing
import argparse

start = time.time()

#hyperparameters
epochs = 120
batch_size = 40
lr = 2e-2
decay = 1e-5
numvids =  1314 #- 1183
valsplit = 0.1
################
vidNamesFileName = 'Vivaset/dynamic_gestures/trainData.txt'
labelsFileName = 'Vivaset/dynamic_gestures/trainLabels.txt'
videosFolder = 'Vivaset/dynamic_gestures/'

weightsFileName = 'weights/weights.h5'

dim_ordering = K.image_dim_ordering()
print ("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(dim_ordering))
backend = dim_ordering

def diagnose(data, verbose=True, label='input', plots=False, backend='tf'):
    # Convolution3D?
    if data.ndim > 2:
        if backend == 'th':
            data = np.transpose(data, (1, 2, 3, 0))
        #else:
        #    data = np.transpose(data, (0, 2, 1, 3))
        min_num_spatial_axes = 10
        max_outputs_to_show = 3
        ndim = data.ndim
        print ("[Info] {}.ndim={}".format(label, ndim))
        print ("[Info] {}.shape={}".format(label, data.shape))
        for d in range(ndim):
            num_this_dim = data.shape[d]
            if num_this_dim >= min_num_spatial_axes: # check for spatial axes
                # just first, center, last indices
                range_this_dim = [0, num_this_dim/2, num_this_dim - 1]
            else:
                # sweep all indices for non-spatial axes
                range_this_dim = range(num_this_dim)
            for i in range_this_dim:
                new_dim = tuple([d] + range(d) + range(d + 1, ndim))
                sliced = np.transpose(data, new_dim)[i, ...]
                print("[Info] {}, dim:{} {}-th slice: "
                      "(min, max, mean, std)=({}, {}, {}, {})".format(
                              label,
                              d, i,
                              np.min(sliced),
                              np.max(sliced),
                              np.mean(sliced),
                              np.std(sliced)))
        if plots:
            # assume (l, h, w, c)-shaped input
            if data.ndim != 4:
                print("[Error] data (shape={}) is not 4-dim. Check data".format(data.shape))
                return
            l, h, w, c = data.shape
            if l >= min_num_spatial_axes or \
                h < min_num_spatial_axes or \
                w < min_num_spatial_axes:
                print("[Error] data (shape={}) does not look like in (l,h,w,c) "
                      "format. Do reshape/transpose.".format(data.shape))
                return
            nrows = int(np.ceil(np.sqrt(data.shape[0])))
            # BGR
            if c == 3:
                for i in range(l):
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                    im = np.squeeze(data[i, ...]).astype(np.float32)
                    im = im[:, :, ::-1] # BGR to RGB
                    # force it to range [0,1]
                    im_min, im_max = im.min(), im.max()
                    if im_max > im_min:
                        im_std = (im - im_min) / (im_max - im_min)
                    else:
                        print ("[Warning] image is constant!")
                        im_std = np.zeros_like(im)
                    plt.imshow(im_std)
                    plt.axis('off')
                    plt.title("{}: t={}".format(label, i))
                plt.show()
                #plt.waitforbuttonpress()
            else:
                for j in range(min(c, max_outputs_to_show)):
                    for i in range(l):
                        mng = plt.get_current_fig_manager()
                        mng.resize(*mng.window.maxsize())
                        plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                        im = np.squeeze(data[i, ...]).astype(np.float32)
                        im = im[:, :, j]
                        # force it to range [0,1]
                        im_min, im_max = im.min(), im.max()
                        if im_max > im_min:
                            im_std = (im - im_min) / (im_max - im_min)
                        else:
                            print ("[Warning] image is constant!")
                            im_std = np.zeros_like(im)
                        plt.imshow(im_std)
                        plt.axis('off')
                        plt.title("{}: o={}, t={}".format(label, j, i))
                    plt.show()
                    #plt.waitforbuttonpress()
    elif data.ndim == 1:
        print("[Info] {} (min, max, mean, std)=({}, {}, {}, {})".format(
                      label,
                      np.min(data),
                      np.max(data),
                      np.mean(data),
                      np.std(data)))
        print("[Info] data[:10]={}".format(data[:10]))

    return

def main():
    show_images = False
    diagnose_plots = False
	plotModel = False
    model_dir = './models'
    global backend

    # override backend if provided as an input arg
    if len(sys.argv) > 1:
        if 'tf' in sys.argv[1].lower():
            backend = 'tf'
        else:
            backend = 'th'
    print ("[Info] Using backend={}".format(backend))

    print("[Info] Reading model architecture...")
    #model = model_from_json(open(model_json_filename, 'r').read())
    model = c3d_model.get_model(backend=backend)
    
    model.summary()
    

    # visualize model
    if plotModel == True:
		model_img_filename = os.path.join(model_dir, 'c3d_model.png')
		from keras.utils import plot_model
		plot_model(model, to_file='pics/newmodel.png')
    

    # Loading model weights 
	print("[Info] Loading model weights...")
	model.load_weights(weightsFileName, by_name=True)
	print("[Info] Loading model weights -- DONE!")
   
	# Model compilation with stochastic gradient descent and metrics	
    sgd = optimizers.SGD(lr, decay)
    model.compile(loss='mean_squared_error', optimizer = sgd, metrics = [metrics.top_k_categorical_accuracy, metrics.categorical_accuracy])
    
	
    #Read filenames to array 
    f = open(str(vidNamesFileName))
    vidnames = f.readlines()
    print(np.shape(vidnames))
	
	#Read labels to array 
	g = open(str(labelsFileName))
    labels = g.readlines()
	
	#Convert to onehot representation
	label_onehot = sklearn.preprocessing.LabelBinarizer()
    label_onehot.fit(range(32))
	labels = label_onehot.transform(labels)
    
	#Directories for logs and plots 
	now = datetime.datetime.now()
	logfolder = 'logs/' +str(numvids) +'_' +str(epochs) +'_' +str(batch_size) +'_' +str(lr) +now.strftime("_%Y-%m-%d_%H%M")
    os.makedirs(logfolder)
    print(logfolder) 
	log = open(str(logfolder) +'/log.txt', 'w')
	vids = [0] * numvids
		
    
    #Loading videos to memory 
	for i in range(numvids):
		print(i)
		vidname = vidnames[i ].replace("\n","").replace("\r","")
		path = "{}{}".format(videosFolder, vidname)
		cap = cv2.VideoCapture(str(path))
		vid = []
		while True:
			ret, img = cap.read()
			#print(np.shape(img))
			if not ret:
				break
			img = misc.imresize(img, (128, 171))
			vid.append(img)
		#print(np.shape(vid))
		
		
		shape = np.shape(vid)
		con = np.zeros((16,128,171,3))
		if shape[0] < 16:
			vid = np.concatenate((vid,con))
			print('Video shorter than 16 frames, zero padded')
			
		shape = np.shape(vid)
		
		
		
		X = np.array(vid, dtype=np.float64)[ 0 : 16 , : , : , : ]
		#diagnose(X, verbose=True, label='X (16-frame clip)', plots=show_images)

		# subtract mean
		mean_cube = np.load('models/train01_16_128_171_mean.npy')
		mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
		#diagnose(mean_cube, verbose=True, label='Mean cube', plots=show_images)
		X -= mean_cube
		#diagnose(X, verbose=True, label='Mean-subtracted X', plots=show_images)

		# center crop
		X = X[:, 8:120, 30:142, :] # (l, h, w, c)
		#diagnose(X, verbose=True, label='Center-cropped X', plots=show_images)
		vids[i] = np.array(X, dtype=np.float32)
	
	
	#Evaluation
	output = model.evaluate(np.array(vids), np.array(labels), batch_size=1)		
	log.write('Evaluation: \n')
	log.write(str(output))
	print(str(output))

	log.write('\nSamples: ' +str(numvids) +' in this validation:' +str(numvids*valsplit) +' Epochs: ' +str(epochs) +' Batch size:  ' +str(batch_size) +' Learning rate: ' +str(lr) +' ')
	log.write('\nWith weights: ' +model_weight_filename)
	log.write(now.strftime("%Y-%m-%d %H:%M"))        	
	log.write(str(output.history))

	
	
	end = time.time()
	print('Time of execution: ' +str((end-start)/60))

if __name__ == '__main__':
    main()
