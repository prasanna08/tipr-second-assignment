import nn as nnet
import numpy as np
import reader
import batch_generators as bgen
import argparse

def print_scores(scores):
    print("Test Accuracy :: %.3f" % (scores['accuracy']))
    print("F1 Micro :: %.3f" % (scores['f1-micro']))
    print("F1 Macro :: %.3f" % (scores['f1-macro']))

def load_mnist_model():
    return nnet.NNet.load('../models/mnist-model')

def load_cat_dog_model():
    return nnet.NNet.load('../models/cat-dog-model')

def load_model(model):
    return load_mnist_model() if model == 'MNIST' else load_cat_dog_model()

def test_model(test_data, nn):
    tx, ty = test_data
    scores = nn.score(tx, ty)
    print_scores(scores)

def train_model(data_path, config):
    x, y = reader.read_dataset(data_path)
    config = [int(i) for i in config[1:-1].split(' ')]
    bg = bgen.BatchGenerator(x, y, 200)
    nn = nnet.NNet(x.shape[1], y.shape[1], config)
    nn.minibatch_train(bg, 5)
    return nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--configuration', type=str)
    args = parser.parse_args()
    
    #config = [int(i) for i in args.configuration[1:-1].split(' ')]
    #print(args.test_data, args.dataset, args.train_data, config)
    
    if args.train_data:
       nn = train_model(args.train_data, args.configuration)
    else:
       nn = load_model(args.dataset)

    test_data = reader.read_dataset(args.test_data, resize=(28, 28) if args.dataset == 'Cat-Dog' else None)
    test_model(test_data, nn)
