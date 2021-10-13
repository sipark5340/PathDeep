# PathDeep

PathDeep is a deep neural network that has biological functional links between gene nodes and pathway nodes which discriminates cancer from normal tissues. The overall scheme of the PathDeep is emphasized in the following figure:


![KakaoTalk_Photo_2021-10-06-14-56-36](https://user-images.githubusercontent.com/51738181/136148176-e36081df-fff0-407b-a564-33ca455b3e0a.png)



More detailed explaination of the PathDeep can be found in the manuscript. 

 

# Data explanation

./data/Toy_data_for_PathDeep.csv

This file consist of 100 cancer tissue and 100 normal tissue and reactome member genes.

./data/Train_Test_Sets/Train, ./data/Train_Test_Sets/Test

There are 20 sets of train and test samples.


# PathDeep example code explanation 
 
When the PathDeep example code is executed, the cancer and normal discrimination performance of PathDeep performed in each hyperparameter is created in ./result/PathDeep_performance.csv.

PathDeep's model structure and weights are stored in the './result' folder.

To run this code, the pre-requisites are as follow: 

Python 2.7

TensorFlow 1.14.0

NumPy 1.17.3

Pandas 0.25.2

Keras 2.3.1



# Extract PathDeep gene pathway index code explanation


```c

code section
...


```
