# PathDeep

PathDeep is a deep neural network that has biological functional links between gene nodes and pathway nodes which discriminates cancer from normal tissues. The overall scheme of the PathDeep is emphasized in the following figure:


![KakaoTalk_Photo_2021-10-06-14-56-36](https://user-images.githubusercontent.com/51738181/136148176-e36081df-fff0-407b-a564-33ca455b3e0a.png)



More detailed explaination of the PathDeep can be found in the manuscript. 

 

# Data explanation

```./data/Toy_data_for_PathDeep.csv```

-> This file consist of 100 cancer tissue and 100 normal tissue and reactome member genes.

```./data/Train_Test_Sets/Train/~.csv```, ```./data/Train_Test_Sets/Test/~.csv```

-> There are 20 sets of train and test samples.

```./data/pathways/~.gmt```

-> There are 15 sets of molecular signature collections that contain biological relationship informations.

# ```PathDeep_example.py``` code block explanation 
 
When the ```PathDeep_example.py``` is executed, the cancer and normal discrimination performance of PathDeep performed in each hyperparameter is created in ```./result/PathDeep_performance.csv.```

PathDeep's model structure and weights are stored in the './result' folder.

To run this code, the pre-requisites are as follow: 

- ```Python 2.7```

- ```TensorFlow 1.14.0```

- ```NumPy 1.17.3```

- ```Pandas 0.25.2```

- ```Keras 2.3.1```


### Source code block #1

```c
gene_idx = {}
for key,values in pathway.items():
    gene_idx[key] = []
    for value in values : 
        if value in raw_dict.keys():
            gene_idx[key].append(raw_dict[value])
```

Generate gene index dictionary 

ex) gene_idx = {EGFR : [1], PLA2G10 : [2], ... } 

- ```pathway``` : dictionary that contains gene to pathway linkage information.
- ```gene_idx``` : gene index dictionary


### Source code block #2

```c
subtrains =[]
subtests = []
for geneset , idx in gene_idx.items():
    if idx :
        subtrain_x = train_x[:, idx]
        subtrains.append(subtrain_x)
        subtest_x = test_x[:, idx]
        subtests.append(subtest_x)
```

This part makes the gene expression matrix as an input shape of PathDeep.
- ```subtrains``` : data set for training PathDeep.
- ```subtests``` : data set for testing PathDeep performance.


### Source code block #3

```c
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
```

This part is for structuring PathDeep

- ```input_items``` : contains gene to pathway linkage list 
- ```geneset_layers``` : pathway layer nodes list
- ```geneset_merged``` : pathway layer (concatanated pathway layer nodes)


### Output result files

PathDeep generates two types of output files: performance and model.

   1. Performance result file
      - ```./result/PathDeep_performance.csv``` : This file contains PathDeep performance (Accuracy, sensitivity, specificity, AUC)
      
      Example)
      
      |index|c2_reactome|...|test_spe|test_auc|
      |------|---|---|---|---|
      |1|c2_reactome|...|0.9927869537946895|0.9952615272462183|
      |2|c2_reactome|...|0.9943550073175831|0.9963550209586295|
      |3|c2_reactome|...|0.9932051014007944|0.9936212866776016|

   2. PathDeep model structure, weight
      - ```./result/~.json``` : This file contains structure of trained PathDeep.
      - ```./result/~.h5``` : This file contains weight of trained PathDeep.




# Extract_PathDeep_gene_pathway_index.py code block explanation

When the Extract_PathDeep_gene_pathway_index.py is executed, user can obtain pathway index and pathway contribution gene index.
These result are saved in below files.

   1. pathway index
      - ```./result/PathDeep_performance.csv```

   2. pathway contribution gene index
      - ```./result/PathDeep_performance.csv```


### Source code block #1

```c
from keras.models import model_from_json 
json_file = open("./result/~.json", "r")

loaded_model_json = json_file.read() 
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("./result/~.h5")
```

This part is for structuring PathDeep

code section


|제목|내용|설명|
|------|---|---|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
