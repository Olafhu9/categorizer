from __future__ import division
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
from keras import optimizers

timer = ROOT.TStopwatch()
timer.Start()

trainInput = "/home/juhee5819/categorizer/ttbar.h5"
data = pd.read_hdf(trainInput)

# delete part of tth events
pd_tth = data.loc[data['category'] == 0]
remove_tth = 1043250 #1181419 #993955
tth_drop_indices = np.random.choice(pd_tth.index, remove_tth, replace=False)
pd_tth_droped = pd_tth.drop(tth_drop_indices)

# delete part of ttbar lf events
np.random.seed(10)
pd_lf = data.loc[data['category'] == 1]
remove_lf = 4294203 #2001232 #1829600
lf_drop_indices = np.random.choice(pd_lf.index, remove_lf, replace=False)
pd_lf_droped = pd_lf.drop(lf_drop_indices)

# delete part of b events
pd_b = data.loc[data['category'] == 2]
remove_b = 121244 #41141
b_drop_indices = np.random.choice(pd_b.index, remove_b, replace=False)
pd_b_droped = pd_b.drop(b_drop_indices)

# bb
pd_bb = data.loc[data['category'] == 3]
remove_bb = 0
bb_drop_indices = np.random.choice(pd_bb.index, remove_bb, replace=False)
pd_bb_droped = pd_bb.drop(bb_drop_indices)

# delete part of c events
pd_c = data.loc[data['category'] == 4]
remove_c = 373517
c_drop_indices = np.random.choice(pd_c.index, remove_c, replace=False)
pd_c_droped = pd_c.drop(c_drop_indices)

# delete part of cc events
pd_cc = data.loc[data['category'] == 5]
remove_cc = 24156
cc_drop_indices = np.random.choice(pd_cc.index, remove_cc, replace=False)
pd_cc_droped = pd_cc.drop(cc_drop_indices)

# merge data
pd_data = pd_tth_droped.append(pd_lf_droped)
pd_data = pd_data.append(pd_b_droped)
pd_data = pd_data.append(pd_bb_droped)
pd_data = pd_data.append(pd_c_droped)
pd_data = pd_data.append(pd_cc_droped)

# pickup only interesting variables
variables = ["ngoodjets","nbjets_m", "nbjets_t", "ncjets_l", "ncjets_m", "ncjets_t", "sortedjet_pt1", "sortedjet_pt2", "sortedjet_pt3", "sortedjet_pt4", "sortedjet_eta1", "sortedjet_eta2", "sortedjet_eta3", "sortedjet_eta4", "sortedjet_phi1", "sortedjet_phi2", "sortedjet_phi3", "sortedjet_phi4", "sortedjet_mass1", "sortedjet_mass2", "sortedjet_mass3", "sortedjet_mass4", "sortedjet_btag1", "sortedjet_btag2", "sortedjet_btag3", "sortedjet_btag4", "deltaR_j12", "deltaR_j34"]
#variables = ["ngoodjets","nbjets_m", "nbjets_t", "ncjets_l", "ncjets_m", "ncjets_t", "sortedjet_pt1", "sortedjet_pt2", "sortedjet_pt3", "sortedjet_pt4", "sortedjet_eta1", "sortedjet_eta2", "sortedjet_eta3", "sortedjet_eta4", "sortedjet_phi1", "sortedjet_phi2", "sortedjet_phi3", "sortedjet_phi4", "sortedjet_mass1", "sortedjet_mass2", "sortedjet_mass3", "sortedjet_mass4", "sortedjet_btag1", "sortedjet_btag2", "sortedjet_btag3", "sortedjet_btag4", "deltaR_j12"]
#variables = ['ncjets_l', 'ncjets_m', 'ncjets_t', 'nbjets_m', 'nbjets_t', 'sortedjet_pt1', 'sortedjet_pt2', 'sortedjet_pt3', 'sortedjet_pt4', 'sortedjet_eta1', 'sortedjet_eta2', 'sortedjet_eta3', 'sortedjeteta_4', 'sortedjet_phi1', 'sortedjet_phi2', 'sortedjet_phi3', 'sortedjet_phi4', 'sortedjet_mass1', 'sortedjet_mass2', 'sortedjet_mass3', 'sortedjet_mass4', 'sortedjet_btag1', 'sortedjet_btag2', 'sortedjet_btag3', 'sortedjet_btag4']
#variables = ["ngoodjets", "nbjets_t", "ncjets_l", "ncjets_m", "ncjets_t", "sortedbjet_pt_1", "sortedbjet_pt_2", "sortedbjet_eta_1", "sortedbjet_eta_2", "sortedbjet_phi_1", "sortedbjet_phi_2", "sortedbjet_mass_1", "sortedbjet_mass_2", "goodjet_pt_1", "goodjet_pt_2", "goodjet_pt_3", "goodjet_pt_4", "goodjet_eta_1", "goodjet_eta_2", "goodjet_eta_3", "goodjet_eta_4", "goodjet_phi_1", "goodjet_phi_2", "goodjet_phi_3", "goodjet_phi_4", "goodjet_mass_1", "goodjet_mass_2", "goodjet_mass_3", "goodjet_mass_4", "goodjet_bD_1", "goodjet_bD_2", "goodjet_bD_3", "goodjet_bD_4", "sortedjet_pt1", "sortedjet_pt2", "sortedjet_pt3", "sortedjet_pt4", "sortedjet_eta1", "sortedjet_eta2", "sortedjet_eta3", "sortedjet_eta4", "sortedjet_phi1", "sortedjet_phi2", "sortedjet_phi3", "sortedjet_phi4", "sortedjet_mass1", "sortedjet_mass2", "sortedjet_mass3", "sortedjet_mass4"]

pd_train_out  = pd_data.filter(items = ['category'])
pd_train_data = pd_data.filter(items = variables)

#covert from pandas to array
train_out = np.array( pd_train_out )
train_data = np.array( pd_train_data )


numbertr=len(train_out)

print numbertr

#Shuffling
order=shuffle(range(numbertr),random_state=100)
train_out=train_out[order]
train_data=train_data[order,0::]

#train_out = train_out.reshape( (1, 3477774) )
#print(train_out)
#train_out = to_categorical( train_out )
#train_out = to_categorical(train_out)

#print(train_out)
trainnb=0.7 # Fraction used for training

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
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(300, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
 # tf.keras.layers.Dense(100, activation=tf.nn.relu),
 # tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])

#modelshape = "10L_300N"
batch_size = 512
epochs = 30
model_output_name = 'model_ttbar_%dE' %(epochs)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy', 'categorical_accuracy'])
hist = model.fit(train_data, train_data_out, batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_data_out))

    #using only fraction of data
    #evaluate = model.predict( valid_data ) 

model.summary()

pred = model.predict(valid_data)
pred = np.argmax(pred, axis=1)
#pred = to_categorical(pred)
comp = np.argmax(valid_data_out, axis=1)

result = pd.DataFrame({"real":comp, "pred":pred})

result_array = []
for i in range(6):
    result_real = result.loc[result['real']==i]
    temp = [(len(result_real.loc[result_real["pred"]==j])) for j in range(6)]
    result_array.append(temp)
    print temp, len(result_real)

result_array_prob = []
for i in range(6):
    result_real = result.loc[result['real']==i]
    temp = [(len(result_real.loc[result_real["pred"]==j])) / len(result_real) for j in range(6)]
    result_array_prob.append(temp)
    print temp, len(result_real)

#np.savetxt("result.csv", result_array + result_array_prob )

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

timer.Stop()
rtime = timer.RealTime(); # Real time (or "wall time")
ctime = timer.CpuTime(); # CPU time
print("RealTime={0:6.2f} seconds, CpuTime={1:6.2f} seconds").format(rtime,ctime)
