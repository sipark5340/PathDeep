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



raw_data=pd.read_csv("./data/Toy_data_for_PathDeep.csv")


K.clear_session()

from keras.models import model_from_json 
json_file = open("./result/PathDeep_structure.json", "r")

loaded_model_json = json_file.read() 
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("./result/PathDeep_weight.h5")


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



# Extract pathway contribution gene index

data_x=np.array(raw_data.iloc[:,1:-1])

subdata = []
for geneset , idx in gene_idx.items():
    if idx :
        subdata_x = data_x[:, idx]
        subdata.append(subdata_x)
        
gene_dict={}
for i in range(len(raw_genes)):
    gene_dict[raw_genes[i]]=raw_genes[i]
    
gene_names = {}
c=0
for key,values in pathway.items():
    gene_names[c] = []
    for value in values : 
        if value in gene_dict.keys():
            gene_names[c].append(gene_dict[value])
    c+=1



pcgi_mat=df()

for i in range(len(pathways)):
    print(i+1)
    print(list(gene_idx.keys())[i])
    dense_name='dense_'+str(i+1)
    sub_weights=model.get_layer(dense_name).get_weights()
    sub_index=df(subdata[i]*sub_weights[0].transpose())
    sub_index.columns=gene_names[i]
    sub_sum=df(sub_index.sum())
    sub_sum.columns=[list(gene_idx.keys())[i]]
    
    pcgi_mat=pd.concat([pcgi_mat,sub_sum],axis=1)

pathway_contribution_gene_index=df(abs(pcgi_mat.sum(axis=1)/len(raw_data))).sort_values([0],ascending=0)
pathway_contribution_gene_index.columns=['index']




# Extract pathway index

pathway_layer=Model(inputs=model.inputs,outputs=model.get_layer('concatenate_1').output)
pathway_index_mat=pathway_layer.predict(subdata)

pathway_index_df=df(pathway_index_mat).transpose()
pathway_index_df.index=pathway.keys()
pathway_index_df.columns=raw_data['sample']

sample_pathway_index_df=(pathway_index_df-pathway_index_df.mean())/pathway_index_df.std()

