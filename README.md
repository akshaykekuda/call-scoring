# CSR-Call-Scoring

DataLoader_fns.py: contains functions needed to create dataloader object (save vocabulary, pad reviews, collate, get indices of a sentence from the vocab)

DatasetClasses.py: contains the classes that define the Yelp dataset and the Call Transcripts dataset

Inference_fns.py: contains functions that make inferences (getting accuracy, predicting for a new data point)

MainCalls.py: creates datasets from call transcripts for train, test, and dev, saves vocabulary, creates dataloaders, creates initial word embeddings matrix, trains model, runs model on test set and reports accuracies, saves the final model after training

MainYelp.py: creates datasets from Yelp Dataset for train, test, and dev, saves vocabulary, creates dataloaders, creates initial word embeddings matrix, trains model, runs model on test set and reports accuracies, saves the final model after training

MainCallsInference.py: loads in saved model and uses it to predict whether a new call is good or bad (cmd line argument is the file name that you want to run inference on)

Models.py: contains class definitions for the encoder and binary classifier pytorch models

Preprocessing.py: contains functions for call transcript preprocessing (takes in call transcript and returns an array of cleaned tokenized sentences)

TrainModel.py: contains function to train the model and reports accuracy and loss at each epoch



Yelp files:

dataset_dev.json: JSON file for the dev set of Yelp reviews. Each record contains the preprocessed review body and its classification. 

dataset_test.json: JSON file for the test set of Yelp reviews. Each record contains the preprocessed review body and its classification. 

dataset_train.json: JSON file for the train set of Yelp reviews. Each record contains the preprocessed review body and its classification. 

yelp_preprocessing.py: Preprocesses the Yelp dataset. Takes the original Yelp dataset and creates dev, test, and train dataset JSON files from a subset. Cleans each review as it makes the dataset. Also reports statistics on the subset of the dataset used. NOTE: this script is not part of the model pipeline. It should be run to produce the three JSON files listed above after any changes are made. This script requires yelp_academic_dataset_review.json, which contains the full dataset of Yelp reviews. This can be downloaded from https://www.kaggle.com/yelp-dataset/yelp-dataset. 
