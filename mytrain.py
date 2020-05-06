from __future__ import division
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from sklearn.utils import shuffle
import re
import string
import math
from ROOT import TFile, TTree
from ROOT import *
import ROOT
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from array import array
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

timer = ROOT.TStopwatch()
timer.Start()

trainInput = "/home/juhee5819/ttbb/ttbar.h5"
data = pd.read_hdf(trainInput)

# delete part of ttbar lf events
np.random.seed(10)
pd_lf = data.loc[data['category'] == 1]
remove_n = 1000000
drop_indices = np.random.choice(pd_lf.index, remove_n, replace=False)
pd_lf_droped = pd_lf.drop(drop_indices)

# merge data
pd_notlf = data.loc[data['category'] != 1]
pd_data = pd_notlf.append(pd_lf_droped)

# pickup only interesting variables
variables = ['ncjets_l', 'ncjets_m', 'ncjets_t', 'nbjets_m', 'nbjets_t', 'sortedjet_pt_1', 'sortedjet_pt_2', 'sortedjet_pt_3', 'sortedjet_pt_4', 'sortedjet_eta_1', 'sortedjet_eta_2', 'sortedjet_eta_3', 'sortedjet_eta_4', 'sortedjet_phi_1', 'sortedjet_phi_2', 'sortedjet_phi_3', 'sortedjet_phi_4', 'sortedjet_mass_1', 'sortedjet_mass_2', 'sortedjet_mass_3', 'sortedjet_mass_4', 'sortedjet_bD_1', 'sortedjet_bD_2', 'sortedjet_bD_3', 'sortedjet_bD_4']
pd_train_out  = pd_data.filter(items = ['category'])
pd_train_data = pd_data.filter(items = variables)

#covert from pandas to array
train_out = np.array( pd_train_out )
train_data = np.array( pd_train_data )

numbertr=len(train_out)

#Shuffling
order=shuffle(range(numbertr),random_state=100)
train_out=train_out[order]
train_data=train_data[order,0::]

#train_out = train_out.reshape( (1, 3477774) )
#print(train_out)
#train_out = to_categorical( train_out )
#train_out = to_categorical(train_out)

#print(train_out)
trainnb=0.8 # Fraction used for training

#Splitting between training set and cross-validation set
valid_data=train_data[int(trainnb*numbertr):numbertr,0::]
valid_data_out=train_out[int(trainnb*numbertr):numbertr]
valid_data_out = to_categorical(valid_data_out)

train_data_out=train_out[0:int(trainnb*numbertr)]
train_data=train_data[0:int(trainnb*numbertr),0::]
train_data_out = to_categorical(train_data_out)

import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
#  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
#  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])

#modelshape = "10L_300N"
batch_size = 10000
epochs = 50
model_output_name = 'model_ttbar_%dE' %(epochs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'categorical_accuracy'])
hist = model.fit(train_data, train_data_out, batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_data_out))

    #using only fraction of data
    #evaluate = model.predict( valid_data ) 

model.summary()

print("Plotting scores")
plt.plot(hist.history['categorical_accuracy'])
plt.plot(hist.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc='lower right')
plt.savefig(os.path.join('fig_score_acc.pdf'))
plt.gcf().clear()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper right')
plt.savefig(os.path.join('fig_score_loss.pdf'))
plt.gcf().clear()


# check the matching efficiency with full dataset 
# filter array with only interesting variables
#jetCombi = data.filter(items = variables)

# convert to array
#input_data = np.array(jetCombi)

# predict from the model
#pred = model.predict( input_data )

# change the format to pandas
#pred = pd.DataFrame(pred, columns=['pred'])

# add prediction as an element
#output_data = pd.concat([data,pred], axis=1)

#calculate total number of events 
#total_selected_events = output_data.groupby('event')
#nEvents = len(total_selected_events)

#calculate machable events 
#matachable = output_data[ output_data['signal'] > 0 ] 
#matachable_grouped = matachable.groupby('event')
#nmatachable = len(matachable_grouped) 

#select the max combination per event
#idx = output_data.groupby('event')['pred'].transform(max) == output_data['pred']
#output_events = output_data[idx]

#select correctly predicted events
#output_events_correct = output_events[ output_events['signal'] > 0 ]

#calcuate the efficiency
#num = len(output_events_correct)
#eff = num / nEvents 
#matchable_ratio = nmatachable / nEvents 

#print
#print "matched : ", num , "/", nEvents , " = ", eff 
#print "matchable : ", nmatachable, "/", nEvents, " = ", matchable_ratio

timer.Stop()
rtime = timer.RealTime(); # Real time (or "wall time")
ctime = timer.CpuTime(); # CPU time
print("RealTime={0:6.2f} seconds, CpuTime={1:6.2f} seconds").format(rtime,ctime)
#print("{0:4.2f} events / RealTime second .").format( nEvents/rtime)
#print("{0:4.2f} events / CpuTime second .").format( nEvents/ctime)
