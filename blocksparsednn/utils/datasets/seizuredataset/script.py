import pandas as pd 
import gzip
import json
import codecs
import os

#constants
ROOT_DIR = "/Users/Ozan Gokdemir/Desktop/dlsystems"
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data.csv")
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "train.jsonl.gz")
VALID_DATA_PATH = os.path.join(ROOT_DIR, "valid.jsonl.gz")
TEST_DATA_PATH = os.path.join(ROOT_DIR, "test.jsonl.gz")


raw_data = pd.read_csv(RAW_DATA_PATH, delimiter=",") #--> https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

raw_data.drop(raw_data.columns[0], axis=1, inplace=True) #no need for ID. 

#Dictionary-fy each line. 
all_data = raw_data.apply(lambda x: {"inputs": x[:-1], "outputs":x[-1] }, axis=1) 

#dataset already comes shuffled, directly slice into train-test-validation 80%,10%,10% respectively.
train_data = all_data[:int(len(all_data)*0.8)]
valid_data = all_data[len(train_data) : len(train_data)+int(len(all_data)*0.1)]
test_data = all_data[int(len(all_data)*0.9):]


#current features are numpy series, which are not json serializable. casting to list and int instead. 
temp = [train_data, valid_data, test_data]
for i in temp: 
    for j in i:
        j["inputs"] = list(j["inputs"])
        j["outputs"] = int(j["outputs"])
             

#helper for writing into the data into jsonl and gzipping.
def write_to_gzip(file_path, data):
    with gzip.open(file_path, "wb") as f:
        writer = codecs.getwriter('utf-8')
        for i in data.keys():
            writer(f).write(json.dumps(data[i]))
            writer(f).write('\n')

#final touch.
write_to_gzip(TRAIN_DATA_PATH, train_data)
write_to_gzip(TEST_DATA_PATH, test_data)
write_to_gzip(VALID_DATA_PATH, valid_data)
