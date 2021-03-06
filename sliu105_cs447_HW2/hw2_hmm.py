########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
import numpy as np
from operator import itemgetter
from collections import defaultdict
from math import log

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        ### Initialize the rest of your data structures here ###
        # dictionary to store all words including rare ones
        self.pre_word_freq = defaultdict(int)
        # dictionary to store word frequencies wl rare ones replaced by UNK
        self.word_freq = defaultdict(float)
        # dictionary to store tag frequencies
        self.tag_freq = defaultdict(float)
        # transition probabilities: p(t_i -> t_j) for all t_i, t_j
        self.transition_prob = defaultdict(float)
        # emission probabilities: P(w_i|t_j) for all w_i, t_j
        self.emission_prob = defaultdict(float)

        self.num_tags = 0
        # initial probability/frequency
        self.init_prob = defaultdict(float)

    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile) # data is a nested list of TaggedWords

        # build pre_word_freq
        for sentence in data:
            for token in sentence:
                self.pre_word_freq[token.word] += 1

        # replace rare word with UNK in data
        for sen in data:
            for token in sen:
                if self.pre_word_freq[token.word] < self.minFreq:
                    token.word = UNK

        # build word_freq, tag_freq, word_tag_freq, and tag_tag_freq dictionaries
        for sentence in data:
            self.init_prob[sentence[0].tag] += 1.0
            self.word_freq[sentence[0].word] += 1.0
            self.tag_freq[sentence[0].tag] += 1.0
            self.emission_prob[(sentence[0].word, sentence[0].tag)] += 1.0
            for i in range(1, len(sentence)):
                self.word_freq[sentence[i].word] += 1.0
                self.tag_freq[sentence[i].tag] += 1.0
                self.emission_prob[(sentence[i].word, sentence[i].tag)] += 1.0
                self.transition_prob[(sentence[i-1].tag, sentence[i].tag)] += 1.0

        # find total number of tags
        self.num_tags = len(self.tag_freq)

        # normalize initial tag probability
        for tag in self.init_prob.keys():
            self.init_prob[tag] /= len(data)

        # build & normalize emission_prob
        for (word, tag) in self.emission_prob.keys():
            self.emission_prob[(word, tag)] /= self.tag_freq[tag]

        # build transition_prob: P(tag1 -> tag2)
        # with add-1 smoothing
        for tag1 in self.tag_freq.keys():
            for tag2 in self.tag_freq.keys():
                self.transition_prob[(tag1, tag2)] = (self.transition_prob[(tag1, tag2)] + 1.0) \
                                                     / (self.tag_freq[tag1] + self.num_tags)



    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            print(senString)
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    # words: a list of word strings
    # return val: a list of tag strings corresponding to words
    def viterbi(self, words):
        # print("Your second task is to implement the Viterbi algorithm for the HMM tagger")
        # returns the list of Viterbi POS tags (strings)
        # return ["NULL"]*len(words) # this returns a dummy list of "NULL", equal in length to words

        # replace rare words by UNK in words
        words_copy = list(words)
        for i in range(len(words_copy)):
            if words_copy[i] not in self.word_freq:
                words_copy[i] = UNK

        # words: word list; tag_list: tag list
        tag_list = list(self.tag_freq.keys())
        # initialize the matrix (rows: tags, cols: word in words)
        # the matrix stores log probabilities
        matrix = np.zeros((len(tag_list), len(words_copy)))

        # fill in the first column of matrix = P_init(t_i) * P(w0, t_i)
        for i in range(len(tag_list)):
            if self.init_prob[tag_list[i]] == 0.0 or self.emission_prob[(words_copy[0], tag_list[i])]\
                    == 0.0:
                matrix[i][0] = -float("inf")
            else:
                matrix[i][0] = log(self.init_prob[tag_list[i]]) + \
                               log(self.emission_prob[(words_copy[0], tag_list[i])])
            # print(matrix[i][0])

        # stores the index of best tags for each cell in matrix starting from second column
        # the first column of back_ptr_idx will be empty, to match the index of matrix
        back_ptr_idx = -np.ones((len(tag_list), len(words_copy)))

        # calculate the rest
        for i in range(1, len(words_copy)):
            # matrix[i][j] = P(w_i|t_j) * max(matrix[i-1][k] * P(w_i|t_k))
            for j in range(len(tag_list)):
                if self.emission_prob[(words_copy[i], tag_list[j])] == 0.0:
                    matrix[j][i] = -float("inf")
                else:
                    # stores the column of candidates for current matrix[i][j]
                    cur_col = np.zeros(len(tag_list))
                    # print(len(tag_list))
                    for k in range(len(tag_list)):
                        # print('k=', k)
                        cur_col[k] = matrix[k][i-1] + log(self.transition_prob[(tag_list[k],
                                                                                    tag_list[j])])

                    best_pre_idx = np.argmax(cur_col)
                    # print(best_pre_idx, ':', cur_col[best_pre_idx])
                    matrix[j][i] = cur_col[best_pre_idx] + log(self.emission_prob[(words_copy[i],
                                                     tag_list[j])])
                    back_ptr_idx[j][i] = best_pre_idx
        # find the largest cumulative probability in last column
        # and fill in the returned list
        best_tags = []
        # the starting point
        best_idx = np.argmax(matrix[:, -1])
        # print(tag_list)
        # print(tag_list[int(best_idx)])
        best_tags.append(tag_list[int(best_idx)])
        # print(len(tag_list))
        for i in range(len(words_copy)-1, 0, -1):
            # update best idx to its previous best index
            best_idx = back_ptr_idx[int(best_idx), i]
            # print('best_idx = ', best_idx)
            best_tags.append(tag_list[int(best_idx)])
        best_tags.reverse()
        # print(best_tags)
        return best_tags







if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
