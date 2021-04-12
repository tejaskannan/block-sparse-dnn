This the Sentiment140 dataset from Kaggle which can be found here. https://www.kaggle.com/kazanova/sentiment140

The original dataset contains 1.600.000 tweets with their associated sentiment labels (0=negative, 2=neural, 4=positive).

We downsized this dataset to train faster. Our version contains 100K tweets for training, 10K for testing, 10K for validation. 

The data contains 120K tweets in total, approximately 50%-50% distribution of positive and negative tweets in train, validation and test sets.

Labels are 0 and 1. 0 for negative, 1 for positive. 

Each tweet is tokenized and padded into sequences of length 32. For instance, the training set has shape (100000, 32). 
