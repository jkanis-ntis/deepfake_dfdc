# -*- coding: utf-8 -*-
"""
Created on 11/4/2020
@author: Zbynek Zajic
"""

import os
from os import path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow import keras

from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans # neumi cosin distance
#import k_means_cosineDistance
#import Kmeans_z_netu
from random import randint, seed

import matplotlib.pyplot as plt
import numpy as np
import scipy

import kaldi_io
import math
from time import strftime

import gc


def loadList(fname):
  d = {}
  f = open(fname)
  for line in f:
    key, val = line.strip().split()
    d[key] = val
  f.close()
  return d

def loadList1(fname):
  d = []
  f = open(fname)
  for line in f:
    val = line.strip()
    d.append(val)
  f.close()
  return d

#
# Load x-vectors
#
print("Loaded x-vectors.")

seed(42)

#------- Create test/train data --------------
# path_xvec_all = 'f:/KORPUSY/DeepFake/v9_DeepFake/exp/xvector_nnet_1a/DeepFake/xvector.1.ark'
#
# ark_xvec = kaldi_io.read_vec_flt_ark(path_xvec_all)
# xvec_all = {}
# for key, mat in ark_xvec:
#   xvec_all[key] = mat
#
# originals = loadList("originals")
# List_diffPitchKaldi = loadList("dists-pitchKaldi")
# originals_JENOM = loadList1("originals_JENOM")
# List_test_JK = loadList1("videos5-val.list")
#
# #najdeme vsechny originaly nalezici do "List_diffPitchKaldi sady"
# orig_diffPitchKaldi = []
# xvec_orig_diffPitchKaldi = np.array([])
# for key in List_diffPitchKaldi:
#     orig_diffPitchKaldi.append(originals[key])
#     X = xvec_all[originals[key][0:-4]]
#     if xvec_orig_diffPitchKaldi.size == 0:
#         xvec_orig_diffPitchKaldi = X
#     else:
#         xvec_orig_diffPitchKaldi = np.vstack((xvec_orig_diffPitchKaldi, X))
#
#
# #nashlukuj orig_diffPitchKaldi do N kategorii dle cosine distance
# N=10
# # --------K-means -  scykyt learnu neumi cosine distance
# #kmeans = KMeans(n_clusters=N, n_init=50, max_iter=100, n_jobs=1)
# #Xtocentre = kmeans.fit_predict(xvec_orig_diffPitchKaldi)
# # --------K-means -  muj ze scykyt learnu s prepsanou cosineDistace
# #kmeans = k_means_cosineDistance.KMeans(n_clusters=N, n_init=10, max_iter=100, n_jobs=1)
# #XtocentreMujCosine = kmeans.fit_predict(xvec_orig_diffPitchKaldi)
# # --------K-means ekvivalent z netu s cosine distance
# Xtocentre = np.empty(xvec_orig_diffPitchKaldi.shape[0])
# Niter = 10
# Suma=[]
# XtoC=[]
# for iter in range(Niter):
#     cc = []
#     centres = []
#     for n in range(N):
#         cc.append(randint(0,xvec_orig_diffPitchKaldi.shape[0]-1))
#         if n==0:
#             centres = xvec_orig_diffPitchKaldi[cc[n], :] # prvotni inicializace centru shluku
#         else:
#             centres= np.vstack((centres, xvec_orig_diffPitchKaldi[cc[n], :]))
#     Xtocentre_old = Xtocentre+1
#     while not math.floor(sum(Xtocentre==Xtocentre_old)/len(Xtocentre)): # opakovat dokud se Kmeans neustali
#         Xtocentre_old = Xtocentre
#         centres, Xtocentre, distances = Kmeans_z_netu.kmeans(np.transpose(np.transpose(xvec_orig_diffPitchKaldi)), centres,  maxiter=100, metric="cosine")
#     Suma.append(sum(distances))
#     XtoC.append(Xtocentre)
# Xtocentre = XtoC[Suma.index(min(Suma))]
#
# #zjisti obsazenost trid  .. IdealniTestTrida  ma priblizne 10% vsech
# IdealniTestTridaDiff = []
# for n in range(N):
#     IdealniTestTridaDiff.append(abs(sum(Xtocentre==n) - 0.1*len(Xtocentre)))
# IdealniTestTrida = IdealniTestTridaDiff.index(min(IdealniTestTridaDiff))
#
#
# #prumerny vektor ideální a neidealni třídy
# prumerIdealniTestTrida = np.array([])
# prumerNEIdealniTestTrida = np.array([])
# for n in range(len(Xtocentre)):
#     X = xvec_all[orig_diffPitchKaldi[n][0:-4]]
#     if Xtocentre[n] == IdealniTestTrida:
#         if prumerIdealniTestTrida.size == 0:
#             prumerIdealniTestTrida = X
#             pocetIdealniTestTrida = 1
#         else:
#             prumerIdealniTestTrida = prumerIdealniTestTrida + X
#             pocetIdealniTestTrida = pocetIdealniTestTrida + 1
#
#     else:
#         if prumerNEIdealniTestTrida.size == 0:
#             prumerNEIdealniTestTrida =X
#             pocetNEIdealniTestTrida = 1
#         else:
#             prumerNEIdealniTestTrida = prumerNEIdealniTestTrida +X
#             pocetNEIdealniTestTrida = pocetNEIdealniTestTrida +1
# prumerIdealniTestTrida = prumerIdealniTestTrida/pocetIdealniTestTrida
# prumerNEIdealniTestTrida = prumerNEIdealniTestTrida/pocetNEIdealniTestTrida
#
# #rozdel vsechny diff na test a train sadu
# labelsTrain = []
# labelsTest = []
# nameTrain = []
# nameTest = []
# dataTrain = np.array([])
# dataTest = np.array([])
# listTest_diffPitchKaldi = []
# for key in List_diffPitchKaldi:
#     X = xvec_all[key[0:-4]]
#
#     # do test sady uloz jen ty, ktere maji cislo tridy kmeans  = IdealniTestTrida
#     if Xtocentre[orig_diffPitchKaldi.index(originals[key])] == IdealniTestTrida:
#         if dataTest.size == 0:
#             dataTest = X
#         else:
#             # print(X.shape[0])
#             dataTest = np.vstack((dataTest, X))
#         labelsTest.append(1)
#         nameTest.append(key)
#
#         # uloz i jeho original
#         X = xvec_all[originals[key][0:-4]]
#         dataTest = np.vstack((dataTest, X))
#         labelsTest.append(0)
#         nameTest.append(originals[key])
#
#     # v opacnem pripade patri do Train sady
#     else:
#         if dataTrain.size == 0:
#            dataTrain = X
#         else:
#             dataTrain = np.vstack((dataTrain, X))
#         labelsTrain.append(1)
#         nameTrain.append(key)
#
#         # uloz i jeho original
#         X = xvec_all[originals[key][0:-4]]
#         dataTrain = np.vstack((dataTrain, X))
#         labelsTrain.append(0)
#         nameTrain.append(originals[key])
#
# # for key in List_test_JK:
# #     if not(key in List_diffPitchKaldi):
# #         if key in originals:
# #             X = xvec_all[originals[key][0:-4]]
# #             if scipy.spatial.distance.cosine(X, prumerIdealniTestTrida) >  scipy.spatial.distance.cosine(X, prumerNEIdealniTestTrida):
# #                 dataTest = np.vstack((dataTest, X))
# #                 labelsTest.append(0)
# #                 nameTest.append(originals[key])
#
# #doplnim test o originaly, tak aby fake bylo cca 5%
# pocetPozadovany = sum(labelsTest)*20 # = pocet faku *20 = 100%
# for key in originals:
#     if len(labelsTest) == pocetPozadovany :
#         break
#     else:
#         #if not(key in List_diffPitchKaldi): - nemuze byt v List_diffPitchKaldi protoze je z listu originalu
#         if not(key in nameTest):
#             if not (key in nameTrain):
#                 if key[0:-4] in xvec_all:
#                     X = xvec_all[key[0:-4]]
#                     #pokud maa mensi distace k Test tride, pridej ho do ni
#                     if scipy.spatial.distance.cosine(X, prumerIdealniTestTrida) < scipy.spatial.distance.cosine(X, prumerNEIdealniTestTrida):
#                         dataTest = np.vstack((dataTest, X))
#                         labelsTest.append(0)
#                         nameTest.append(key)
#                 else:
#                     print('Pozor:  ' + key + ' nema Xvector!!!\n')
#
#
# np.savetxt('dataTrain_ZZset.csv', dataTrain, delimiter=',')
# np.savetxt('labelsTrain_ZZset.csv', np.asarray(labelsTrain), delimiter=',')
# np.savetxt('dataTest_ZZset.csv', dataTest, delimiter=',')
# np.savetxt('labelsTest_ZZset.csv', np.asarray(labelsTest), delimiter=',')
# with open('nameTest_ZZset.txt', "w") as output:
#     for n in nameTest:
#         output.write(n + '\n')
# with open('nameTrain_ZZset.txt', "w") as output:
#     for n in nameTrain:
#         output.write(n + '\n')


#------- Load train/test data --------------

num_classes = 2

dataTrainDev = np.loadtxt('dataTrain_ZZset.csv', delimiter=',').astype('float32')
labelsTrainDev =  keras.utils.to_categorical(np.loadtxt('labelsTrain_ZZset.csv', delimiter=','), num_classes)
nameTrainDev = loadList1('nameTrain_ZZset.txt')


dataTest = np.loadtxt('dataTest_ZZset.csv', delimiter=',').astype('float32')
labelsTest = keras.utils.to_categorical(np.loadtxt('labelsTest_ZZset.csv', delimiter=','), num_classes)
nameTest = loadList1('nameTest_ZZset.txt')

# chyba=0
# for i in range(len(nameTrainDev)):
#     for j in range(len(nameTest)):
#         if nameTrainDev[i] == nameTest[j]:
#             chyba = chyba+1
#print('Chyba:' + str(chyba) + '\n' )


#-------------------------------------------------------------------------------------
# -----------------data for NN classifier... ivectors--------------------------------------------
#-------------------------------------------------------------------------------------

print("Cross-validating KERAS NN classifier... on x-vectors")

ARD = 'f:/KORPUSY/DeepFake/NNAudioClassifier_xvec_ZZset_2Layers/'
try:
    os.stat(ARD)
except:
    os.mkdir(ARD)
model_path = (ARD + 'NNivec_model_2Layers_{ep}epochs{ba}batch{do}dropout{u}units')
#fig = plt.figure()


n_split=10

#-------------------------------------------------------------------------------------
# -----------------NN classifier... ivectors--------------------------------------------
#-------------------------------------------------------------------------------------
Units_s = [8000]
Dropout_s = [0.5]
Batch_size_s = [256]
Epochs_s = [4,5,6,7,8,10,11,12,13,14,15,16,17,18]

fileResAll = open(ARD + '_2Layers_CrossVal_Results_all.txt', "w")
fileResAll.write('#TrainDev data TRUE: ' + str(sum(labelsTrainDev == 1)) + ',  #TrainDev data FALSE: ' + str(sum(labelsTrainDev == 0)) + '\n')
fileResAll.write('#Test data TRUE:' +  str(sum(labelsTest == 1)) + ', #Test data FALSE:' +  str(sum(labelsTest == 0)) + '\n')
fileResAll.close()

fileResAll_CSV = open(ARD + '_2Layers_CrossVal_Results_all.csv', "w")
fileResAll_CSV.write('units,dropout,batch_size,epochs,CrossvalMean_Dev_Loss,CrossvalMean_Dev_Acc,CrossvalMean_Test_Loss,Crossval_Mean_Test_Acc\n')
fileResAll_CSV.close()

for units in Units_s:
    for dropout in Dropout_s:
        for batch_size in Batch_size_s:
            for epochs in Epochs_s:


                fileResAll = open(ARD + '_2Layers_CrossVal_Results_all.txt', "a")
                fileResAll_CSV = open(ARD + '_2Layers_CrossVal_Results_all.csv', "a")

                train_model_path = model_path.format(ep=epochs, ba=batch_size, do=dropout, u=units) + '.h5'
                train_model_path_graph = model_path.format(ep=epochs, ba=batch_size, do=dropout, u=units) + '_model.png'

                train_model_path_results = model_path.format(ep=epochs, ba=batch_size, do=dropout,  u=units) + '_results.txt'
                train_model_path_fig_loss = model_path.format(ep=epochs, ba=batch_size, do=dropout,  u=units) + '_fig_loss.png'
                train_model_path_fig_acc = model_path.format(ep=epochs, ba=batch_size, do=dropout,  u=units) + '_fig_acc.png'

                # fileVysl = open(train_model_path_results, "w")
                # fileVysl.write('Epochs:' + str(epochs) + '\n')
                # fileVysl.write('Batch_size:' + str(batch_size) + '\n')
                # fileVysl.write('Dropout:' + str(dropout) + '\n')
                # fileVysl.write('units:' + str(units) + '\n\n')

                fileResAll.write('\n''\n''----------- ' + train_model_path + '----------------' + '\n')
                fileResAll.write('Epochs:' + str(epochs) + ', Batch_size:' + str(batch_size) + ', Dropout:' + str(
                    dropout) + ', Units:' + str(units) + '\n')

                fileResAll_CSV.write(str(units) + ',' + str(dropout)  + ',' +   str(batch_size)  + ',' +  str(epochs) + ',')

                #Cross-validation
                score_all=[0,0,0] #acumulovane score pres celou crossvalidaci =[loss, acc, pocet]
                scoreDev_all=[0,0,0] #acumulovane score pres celou crossvalidaci =[loss, acc, pocet]

                for train_index, test_index in KFold(n_split).split(dataTrainDev):
                    x_train, x_dev = dataTrainDev[train_index], dataTrainDev[test_index]
                    y_train, y_dev = labelsTrainDev[train_index], labelsTrainDev[test_index]


                    #fileResAll.write('#Train data TRUE: ' + str(sum(y_train == 1)) + ',  #Train data FALSE: ' + str(sum(y_train == 0) )+ '\n')
                    #fileResAll.write('#Dev data TRUE: ' + str(sum(y_dev == 1)) + ',  #Dev data FALSE: ' + str(sum(y_dev == 0)) + '\n')

                    # print('x_train shape:', x_train.shape)
                    # print('x_dev shape:', x_dev.shape)
                    # print('x_test shape:', x_test.shape)
                    # print('x_testZero shape:', x_testZero.shape)


                    print("Train KERAS NN classifier... on x-vectors")



                    model = Sequential()  # uz neni jina moznost modelu
                    model.add(Dense(units=units, activation='tanh', input_dim=x_train.shape[1], name='dense_1'))
                    model.add(Dropout(dropout, name='dropout_1'))
                    model.add(BatchNormalization())
                    model.add(Dense(units=units, activation='tanh', input_dim=x_train.shape[1], name='dense_2'))
                    model.add(Dropout(dropout, name='dropout_2'))
                    model.add(BatchNormalization())
                    model.add(Dense(num_classes, activation='softmax', name='dense_3'))


                    #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])  # nstaveni ucici algoritmus

                    #########reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=0.000001)

                    # keras.utils.plot_model(model, to_file=train_model_path_graph, show_shapes=True, show_layer_names=True)
                    # print(model.summary())


                    history = model.fit(x_train, y_train,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=0,
                                        #######callbacks=[reduce_lr],
                                        #validation_split=0.1)
                                        validation_data=(x_dev, y_dev))
                                        # natrenuj  .. v priade nevejde do mpameti ...  misto fit train_on_batch (nutne zabespecit nastaveni trenovani)

                    scoreDev = model.evaluate(x_dev, y_dev, verbose=0)  # vypocitej
                    #print('Dev loss:', scoreDev[0])
                    #print('Dev accuracy:', scoreDev[1])

                    score = model.evaluate(dataTest, labelsTest, verbose=0)  # vypocitej
                    #scoreZero = model.evaluate(x_testZero, y_testZero, verbose=0)  # vypocitej
                    #print('Test loss:', score[0])
                    #print('Test accuracy:', score[1])
                    #print('TestZero loss:', scoreZero[0])
                    #print('TestZero accuracy:', scoreZero[1])

                    #fileVysl.write('Test loss:' +  str(score[0]) + '\n')
                    #fileVysl.write('Test accuracy:' + str(score[1])+ '\n\n')
                    #fileVysl.write('TestZero loss:' +  str(scoreZero[0]) + '\n')
                    #fileVysl.write('TestZero accuracy:' + str(scoreZero[1]) + '\n\n')

                    #fileVysl.close()

                    #fileResAll.write('Dev loss:' +  str(scoreDev[0]) + ', Dev accuracy:' + str(scoreDev[1])+ '\n')
                    #fileResAll.write('Test loss:' +  str(score[0]) + ', Test accuracy:' + str(score[1])+ '\n')
                    #fileResAll.write('---------------------------------------' + '\n\n')

                    score_all[0] = score_all[0] + score[0]
                    score_all[1] = score_all[1] + score[1]
                    score_all[2] = score_all[2] + 1

                    scoreDev_all[0] = scoreDev_all[0] + scoreDev[0]
                    scoreDev_all[1] = scoreDev_all[1] + scoreDev[1]
                    scoreDev_all[2] = scoreDev_all[2] + 1

                    # Destroys the current Keras graph and creates a new one.
                    # Useful to avoid clutter from old models / layers.
                    keras.backend.clear_session()
                    gc.collect()
                    del model

                    #model.save(train_model_path)  # creates a HDF5 file 'my_model.h5'
                    #
                    # # list all data in history
                    # print(history.history.keys())
                    # fig.clf()
                    # # summarize history for accuracy
                    # plt.plot(history.history['accuracy'])
                    # plt.plot(history.history['val_accuracy'])
                    # plt.title('model3 accuracy')
                    # plt.ylabel('accuracy')
                    # plt.xlabel('epoch')
                    # plt.legend(['train', 'test'], loc='upper left')
                    # #plt.show()
                    # plt.savefig(train_model_path_fig_acc)
                    #
                    # # summarize history for loss
                    # fig.clf()
                    # plt.plot(history.history['loss'])
                    # plt.plot(history.history['val_loss'])
                    # plt.title('model3 loss')
                    # plt.ylabel('loss')
                    # plt.xlabel('epoch')
                    # plt.legend(['train', 'test'], loc='upper left')
                    # #plt.show()
                    # plt.savefig(train_model_path_fig_loss)

                print('Crossval Mean Dev - Loss:' + str(scoreDev_all[0] / scoreDev_all[2]) + ', Accuracy:' + str(scoreDev_all[1] / scoreDev_all[2]) + '\n')
                print('Crossval Mean Test - Loss:' + str(score_all[0] / score_all[2]) + ', Accuracy:' + str(score_all[1] / score_all[2]) + '\n')
                print('---------------------------------------' + '\n')

                fileResAll.write('Crossval Mean Dev - Loss:' + str(scoreDev_all[0]/scoreDev_all[2]) + ', Accuracy:' + str(scoreDev_all[1]/scoreDev_all[2]) + '\n')
                fileResAll.write('Crossval Mean Test - Loss:' + str(score_all[0] / score_all[2]) + ', Accuracy:' + str(score_all[1] / score_all[2]) + '\n')
                fileResAll.write('---------------------------------------' + '\n')
                #fileResAll.write('---------------------------------------' + '\n\n')
                fileResAll.close()

                fileResAll_CSV.write(str(scoreDev_all[0]/scoreDev_all[2]) + ',' + str(scoreDev_all[1]/scoreDev_all[2]) + ',' + str(score_all[0] / score_all[2])  + ',' + str(score_all[1] / score_all[2]) + '\n')
                fileResAll_CSV.close()
