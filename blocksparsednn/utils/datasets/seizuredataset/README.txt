11/12/2020
Ozan Gokdemir 

This is the Epileptic Seizure Recognition Data Set from UCI ML Repository. 
The dataset can be found here: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

OVERVIEW OF THE DATA: 
Dataset consists of 11500 datapoints. I split it into, 
9200 training data points (80%), 
1150 validation data points (10%), 
1150 test data points (10%). 

Each data point is a one-second-long EEG reading of a patient. There are 500 patients in total and each reading contains 178 numeric features. 
A possible use of data is training a multivariate classifier. Based on the given readings, the model has to classify the entry into following categories;

5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open
4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed
3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area
2 - They recorder the EEG from the area where the tumor was located
1 - Recording of seizure activity

Alternatively, the task can be constructed as a binary classification where labels 2,3,4 and 5 correspond to "no seizure", and label 1 correspond to "recording of seizure". 

DATASET FORMAT:
Each dataset is processed into JSON Line (jsonl) files and gzipped. Decompressed versions
can be found in the "decompressed" folder in case one wants to verify the format.
Each line consists of a Python-type dictionary, inputs: [<list of 178 integers>], outputs: <int 1-5 corresponding to label>. 

FILE STRUCTURE: 
* data.csv --> contains the raw data. 
* script.py --> preprocessing script. 
* decompressed --> contains the ungzipped jsonl files for test, train, validation.
* train --> gzipped training data.
* test --> gzipped testing data.
* valid --> gzipped validation data.
