#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import csv
from pandas import DataFrame as df
import numpy as np
import pickle
from sklearn import metrics

import tensorflow as tf
import keras 
from keras.layers import Input, Dense    #using set model component
from keras.models import Model    #using set model 
from keras import layers as Layer
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization

def gmt_to_dictionary(filename):
    geneset = open(filename, 'r')
    lines = geneset.readlines()
    geneset.close()
    gene_names = []
    genes = []
    for line in lines:
        geneset = line.split('\t')
        gene_names.append(geneset[0])
        if geneset[-1][-1] == '\n':
            geneset[-1] = geneset[-1][:-1]
        genes.append(geneset[2:])

    gene_sets = dict(zip(gene_names, genes))
    return gene_sets

def check_correct(predict, y):
    result = {}
    result['True-Positive'] = 0
    result['True-Negative'] = 0
    result['False-Negative'] = 0
    result['False-Positive'] = 0

    for i in range(len(predict)) :
        if predict[i] == y[i] :
            if y[i] == 0 :
                result['True-Negative'] += 1
            else :
                result['True-Positive'] += 1
        else :
            if y[i] == 0 :
                result['False-Positive'] += 1
            else :
                result['False-Negative'] += 1

    for result_k, result_v in result.items():
        print(result_k +" : "+ str(result_v))
    
    acc=(result['True-Positive']+result['True-Negative'])/len(y)
    sensitivity=result['True-Positive']/(result['True-Positive']+result['False-Negative'])
    specificity=result['True-Negative']/(result['True-Negative']+result['False-Positive'])
    print("Sensitivity :", sensitivity)
    print("Specificity :", specificity)
    return acc, sensitivity, specificity


raw_data=pd.read_csv("./data/Toy_data_for_PathDeep.csv")


pathway_lst=['c2_reactome']#,'c2_kegg','c4_cgn','c1_positional', 'c2_cgp', 'c2_cp_biocarta', 'c2_cp', 'c3_mir', 'c3_tft','c4_cm', 'c5_GO_bp', 'c5_GO_cc', 'c5_GO_mf', 'c6_oncogenic_signatures', 'c7_immunologic_signatures']

pathspace = './data/pathways/'
suffix='_symbols'

raw_data=raw_data.sample(frac=1)
raw_genes = raw_data.columns.values.tolist()
raw_genes = raw_genes[1:-1]
raw_dict = {}

for i in range(len(raw_genes)):
    raw_dict[raw_genes[i]] = i

for p_lst in pathway_lst:
    
    pathway_f = pathspace+p_lst+suffix+'.gmt'
    pathway = gmt_to_dictionary(pathway_f)

    # Generate raw gene index dictionary
    gene_idx = {}
    for key,values in pathway.items():
        gene_idx[key] = []
        for value in values : 
            if value in raw_dict.keys():
                gene_idx[key].append(raw_dict[value])

    layers_list=[]
    for i in range(10): # 6 set
        layers=[500,500,200,300,200,100]
        a_layers=[]
        for node in layers:
            rate=1-i*0.005
            a_layers.append(int(node*rate))
        layers_list.append(a_layers)

    early_stopping_patient_list=[10, 15]
    batch_size_list=[290, 295, 300, 305, 310]
    
    my_col  = ['Index',
               'Pathway_Selection',
               'Layer',
               'Batch_size',
               'Drop-Out',
               'Learning_Rate',
               'Early_stopping_patient',
               'Train_Accuracy',
               'Train_Sensitivity',
               'Train_Specificity',
               'Train_AUC',
               'Test_Accuracy',
               'Test_Sensitivity',
               'Test_Specificity',
               'Test_AUC']

    output_file = "./result/PathDeep_performance.csv"

    csv_out_file = open(output_file, 'w', newline = '')
    filewriter = csv.writer(csv_out_file)
    filewriter.writerow(my_col)
    csv_out_file.close()


    for index in range(1,21):

        train_index=pd.read_csv("./data/Train_Test_Sets/Train/train_"+str(index)+".csv")
        test_index=pd.read_csv("./data/Train_Test_Sets/Test/test_"+str(index)+".csv")
        train_data=df(raw_data[raw_data['sample'].isin(train_index['sample'])])
        test_data=df(raw_data[raw_data['sample'].isin(test_index['sample'])])
        train_x=np.array(train_data.iloc[:,1:-1])
        test_x=np.array(test_data.iloc[:,1:-1])

        train_y=train_data['result'].tolist()
        test_y=test_data['result'].tolist()

        subtrains =[]
        subtests = []
        for geneset , idx in gene_idx.items():
            if idx :
                subtrain_x = train_x[:, idx]
                subtrains.append(subtrain_x)
                subtest_x = test_x[:, idx]
                subtests.append(subtest_x)

        for batch_size in batch_size_list:
            for esp in early_stopping_patient_list:
                for nodes_list in layers_list:

                    K.clear_session()

                    count = 0
                    input_items = []
                    for geneset , idx in gene_idx.items():
                        if idx :
                            input_items.append(Input(shape=(len(subtrains[count][0]),), name = geneset))
                            count += 1

                    geneset_layers = []

                    for input_item in input_items :
                        geneset_layers.append(Dense(1 , activation='relu')(input_item))

                    geneset_merged = Layer.concatenate(geneset_layers, axis=-1)

                    i+=1
                    hiddens = []
                    count = 0

                    for node in nodes_list :
                        if count == 0 :
                            dense = (Dense(node, activation='relu')(geneset_merged))
                            hiddens.append(Dropout(0.5)(dense))
                        else : 
                            dense = Dense(node, activation='relu')(hiddens[count-1])
                            hiddens.append(Dropout(0.5)(dense))

                        count +=1

                    predictions = Dense(1, activation='sigmoid', name='predictions')(hiddens[-1])
                    model = Model(inputs = input_items, output = predictions)

                    adam = keras.optimizers.Adam()
                    model.compile(optimizer=adam, loss ='binary_crossentropy', metrics=['accuracy'])

                    train_loss_best = 100 # for saving best loss value 
                    train_acc_best = 0
                    best_m_model=[] #for saving best model
                    count=0

                    while 1:
                        mo=model.fit(x=subtrains, y=train_y, epochs = 1,batch_size=batch_size)
                        train_loss=mo.history['loss'][0]

                        if train_loss < train_loss_best: # new best model. count reset.
                            train_loss_best = train_loss
                            count=0
                            best_model = model
                        if count>esp: # no increase, stop.
                            model = best_model
                            break
                        else: count=count+1

                    model_json = model.to_json()
                    with open("./result/"+p_lst+
                              "_index_"+str(index)+
                              "_node_"+str(nodes_list[0])+
                              "_batch_size_"+str(batch_size)+
                              "_early_stopping_patient_"+str(esp)+
                              ".json", "w") as json_file : 
                        json_file.write(model_json)

                    model.save_weights("./result/"+p_lst+
                                       "_index_"+str(index)+
                                       "_node_"+str(nodes_list[0])+
                                       "_batch_size_"+str(batch_size)+
                                       "_early_stopping_patient_"+str(esp)+".h5")
                    
                    print("Saved model to disk")

                    train_hypo=model.predict(subtrains)
                    train_pred = np.where(train_hypo > 0.5, 1, 0).flatten()
                    train_acc, train_sen, train_spe=check_correct(train_pred, train_y)
                    train_auc=metrics.roc_auc_score(train_y, train_hypo)

                    test_hypo=model.predict(subtests)
                    test_pred = np.where(test_hypo > 0.5, 1, 0).flatten()
                    test_acc, test_sen, test_spe=check_correct(test_pred, test_y)
                    test_auc=metrics.roc_auc_score(test_y, test_hypo)
    
                    temp_output = [index, p_lst, nodes_list, batch_size, 0.5, 0.001, esp, train_acc, train_sen, train_spe, train_auc, test_acc, test_sen, test_spe, test_auc]
                    csv_out_file = open(output_file, 'a', newline = '')
                    filewriter = csv.writer(csv_out_file)
                    filewriter.writerow(temp_output)
                    csv_out_file.close()
