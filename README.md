# PathDeep

PathDeep is a deep neural network that has biological functional links between gene nodes and pathway nodes, and discriminates cancer and normal tissue.


##### 

# Data explanation

./data/Toy_data_for_PathDeep.csv file consists of 100 cancer tissue and normal tissue, and contains reactome member genes.
There are 20 sets of train and test samples in ./data/Train_Test_Sets/Train/ folder and ./data/Train_Test_Sets/Test/ folder respectively.


# Code explanation 
 
When the PathDeep example code is executed, the cancer and normal discrimination performance of PathDeep performed in each hyperparameter is created in ./result/PathDeep_performance.csv.

PathDeep's model structure and weights are stored in the './model/structure' folder and the './model/weight' folder, respectively.
