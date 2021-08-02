# AI-Final-Project
Final project in Prof. Louzoun's Artificial Intelligence course.
## Abstract
Mushrooms data set was analyzed using artificial intelligence methods to classify the samples by their odor. Three approaches were used. The first approach was to cluster the data, so that each cluster represents a different odor. The second approach was to build a machine that learns the mushroom family based on its features. The third approach was to repeat the other two techniques after creating better features. The goal was to find the best approach among this techniques. Out of the three approaches tested, applying the second approach with SVM RBF kernel with a box constraint of 0.8 was found to be the best with an average F1-Score Score of 0.86449. Then, four approaches for imputing missing values were used on the samples with the missing data of the mushrooms. The imputed data was classified by SVM RBF kernel with a box constraint of 0.8 which found to be the best classification method for the data. Comparison of the results revealed that imputing the missing values with the median of their column is the best imputation approach. At least, the fit between the classification and the real labels was measured and the average F1-Score found to be 0.51726. 

## Data Sets
The data may be found here:
1. [Data](https://github.com/roysgitprojects/AI-Final-Project/blob/main/mushrooms_data.txt)
2. [Data's README](https://github.com/roysgitprojects/AI-Final-Project/blob/main/mushrooms_readme.txt)
3. [Data with missing values](https://github.com/roysgitprojects/AI-Final-Project/blob/main/mushrooms_data_missing.txt)

## Instructions
Running the python code for the Unsupervised Methods may be done directly from [main_file.py](main_file.py).
For the Supervised classification scripts there are instructions bellow.
Than for the data with the missing values one need to run [part_2_main_file.py](part_2_main_file.py).

Note: Running the files must be don according to that order since some of the methods requires files created in oder
 methods (e.g. some of the classification methods uses the dimension reduction results - as detailed in the pdf paper). 

## Instructions for classification methods
By running classification you can generate all of our classifications methods.
When you ran the file you will be asked to enter the dimension reduction method of your choosing, if you wish to use no dimension reduction methods you can just write normal.
The program then will create your classifications additionally, it will run statistical tests between all the classifications to find the best features for a given method and also to find the best method one.

Important note: To save the results from the classification (for example the f1 score) you need to create appropriate directories.
To make it easy one can just open one of the classification folders and copy its content.
By doing that he will be able to run each dimension method he will want (that is given he have the dimension reduction file).


creating_the_plot:

To create the classifications plot by yourself, one will only need to run the file creating_the_plot.

best champion: 

To find the best classification method before and after dimension reduction, one will only need to run the file best champion. 

## Python Modules
The main modules used on this project are:
 * Sklearn
 * Matplotlib
 * Skfuzzy
 * Numpy
 * Pandas
 * Scipy
 * Yellowbrick
 * Torch
 * Xgboost
 * NetworkX