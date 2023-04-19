# TransUFold
A novel deep learning approach for enhancing the accuracy of RNA secondary structure prediction
# Pre-trained models
Pretrained models can be found in our models folder
# data
The data we used in the paper can be found in the data folder
# Usage
## Data generator
You can put the bpseq file into your own folder and run the following code to preprocess the data:
sequences
The sequence length is less than 512：

`python process_data_newdataset.py your own folder`

The sequence length is greater than 512 and less than 1600：

`python process_data_1600.py your own folder`

## Train
You can put the preprocessed cPickle file into the data folder or download the cPickle file we use, and then change the train_files in utils.py in the TU folder to the cPickle you want to train. Then run the following code to train

The sequence length is less than 512：

`python train.py`

The sequence length is greater than 512 and less than 1600：

`python train_1600.py`

## Test
You can download our trained models or train your own models. Put the model in the models folder and the cPickle file to be tested in the data folder. Run the following code to test

The sequence length is less than 512：

`python test.py`

The sequence length is greater than 512 and less than 1600：

`python test_1600.py`

After the test is completed, each predicted bpseq file will be generated in the seq folder, and a csv file will be generated with the test results of each sequence

## Predict
You can use our tools to predict sequences, just put one or more seq files you want to predict (each seq file can only have one sequence, the first line in the file is the RNA name, and the second line is the sequence) into your In your own folder, and then run the following code to predict (provided that you have trained the model yourself or downloaded our model and put it in the models folder)：

`python predict.py your own folder`


It can predict RNAs with a length less than 1600, and a bpseq file will be generated in the results folder after the prediction is completed
