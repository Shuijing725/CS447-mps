########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
import numpy as np

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]

# A class for evaluating POS-tagged data
class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        # nested list of TaggedWord objects
        self.gold_data = self.readLabeledData(goldFile)
        self.test_data = self.readLabeledData(testFile)

        # count number of tags and store all tags in a list
        self.num_data = 0
        self.tags = []
        for sentence in self.gold_data:
            for token in sentence:
                self.num_data += 1
                if token.tag not in self.tags:
                    self.tags.append(token.tag)

        # print(self.tags)
        # conf_matrix: stores the confusion matrix as np array
        self.conf_matrix = np.zeros((len(self.tags), len(self.tags)), dtype = int)
        # fill the confusion matrix (row: correct tags, col: assigned tags)
        for i in range(len(self.gold_data)):
            for j in range(len(self.gold_data[i])):
                row_idx = self.tags.index(self.gold_data[i][j].tag)
                # print(self.test_data[i][j].tag)
                col_idx = self.tags.index(self.test_data[i][j].tag)
                self.conf_matrix[row_idx, col_idx] += 1

    # copied from hw2_hmm.py
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    @staticmethod
    def readLabeledData(inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r")  # open the input file in read-only mode
            sens = []
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence)  # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit()  # exit the script

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        correct = 0.0
        for i in range(len(self.tags)):
            correct += self.conf_matrix[i, i]
        return correct / self.num_data * 1.0


    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        correct = 0.0
        for i in range(len(self.gold_data)):
            cur_correct = True
            for j in range(len(self.gold_data[i])):
                if self.gold_data[i][j].tag != self.test_data[i][j].tag:
                    cur_correct = False
            if cur_correct:
                correct += 1
        return correct / len(self.gold_data) * 1.0

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        # write confusion matrix to file
        f = open(outFile, "w")
        f.write('   '.join(self.tags) + '\n') # first line
        for i in range(len(self.conf_matrix)):
            line = self.conf_matrix[i].astype(str)
            f.write(self.tags[i] + '    ' + '   '.join(line) + '\n')
        f.close()


    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float
    #Return the tagger's precision when predicting tag t_i
    ################################
    def getPrecision(self, tagTi):
        idx = self.tags.index(tagTi)
        return self.conf_matrix[idx, idx] / np.sum(self.conf_matrix[:, idx]) * 1.0

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    # Return the tagger's recall for correctly predicting gold tag t_j
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        idx = self.tags.index(tagTj)
        return self.conf_matrix[idx, idx] / np.sum(self.conf_matrix[idx]) * 1.0


if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and out.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        # print("Token accuracy: ", eval.getTokenAccuracy())
        # print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # # Calculate recall and precision
        # print("Recall on tag NNP: ", eval.getPrecision('NNP'))
        # print("Precision for tag NNP: ", eval.getRecall('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("conf_matrix.txt")
