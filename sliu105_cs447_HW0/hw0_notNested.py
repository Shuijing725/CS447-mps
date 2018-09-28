########################################
## CS447 Natural Language Processing  ##
##           Homework 0               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Read in a text file (consisting of one sentence per line) into a data structure
##
import os.path
import sys
from operator import itemgetter
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus 
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("reading file ", f)
        for line in file:
            i += 1
            sentence = line.split()  # split the line into a list of words
            corpus.extend(sentence)  # extend the current list of words with the words in the sentence
            if i % 1000 == 0:
                sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it
        return corpus
    else:
        print("Error: corpus file ", f, " does not exist")  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit() # exit the script

#-------------------------------------------
# Data output
#-------------------------------------------

# part (a)
def countWords(corpus):
    return len(corpus)

# part (b)
def getVocab(corpus):
    vocab = set(corpus)
    return sorted(vocab)

# part(c)
def printWordFrequencies(index, vocab):
    # create a dictionary of words and their frequencies
    freq_dict = {}
    for key, item in index.iteritems():
        freq_dict[key] = len(item)

    freq_dict_sorted = sorted(freq_dict.items(), key = itemgetter(1), reverse = True)
    
    for word in freq_dict_sorted:
        print("Word: ", word[0], ", frequency: ", word[1])

#
# Print out all occurrences of the word 'word' in the corpus indexed by index
#
def printCorpusConcordance(word, corpus, index):
    # iterate through the corpus to find all occurrences of word
    for i in range(len(corpus)):
    	if corpus[i] == word:
    		printConcordance(corpus, i)

#
#  Print out the concordance of the word at position word_i
#  in sentence sentence, e.g: 
# 
"""
>>> printConcordance(1, ["what's", 'the', 'deal', '?'])
                                  what's    the     deal ?
>>> printConcordance(1,['watch', 'the', 'movie', 'and', '"', 'sorta', '"', 'find', 'out', '.', '.', '.'])
                                   watch    the     movie and " sorta " 
>>> printConcordance(3,['so', 'what', 'are', 'the', 'problems', 'with', 'the', 'movie', '?'])
                             so what are    the     problems with the movie ?     
"""

def printConcordance(corpus, word_i):
    """ print out the five words preceding word,
        the word at position i and the following five words."""
    if word_i < len(corpus):
        start = max(word_i-5, 0)
        end = min(word_i+6, len(corpus))
        left = ' '.join(corpus[start:word_i])
        right = ' '.join(corpus[word_i+1:end])
        print(left.rjust(40), corpus[word_i].center(10), right.ljust(30))

#--------------------------------------------------------------
# Corpus analysis
#--------------------------------------------------------------

#
# Create an index that maps each word to all its positions in the corpus
#
def createCorpusIndex(corpus):
    # we create a dictionary (associative array) that maps words
    # to a list of their positions
    index = {}
    for i in range(len(corpus)):
        word = corpus[i]
        if (word in index):
            index[word] += [i]
        else:
            index[word] = [i]
    return index

# part (a)
def countWords(corpus):
    return len(corpus)

# part (b)
def getVocab(corpus):
    vocab = set(corpus)
    return sorted(vocab)

# part(c)
def printWordFrequencies(index, vocab):
    # create a dictionary of words and their frequencies
    freq_dict = {}
    for key, item in index.items():
        freq_dict[key] = len(item)

    freq_dict_sorted = sorted(freq_dict.items(), key = itemgetter(1), reverse = True)
    
    for word in freq_dict_sorted:
        print("Word: ", word[0], ", frequency: ", word[1])


#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    movieCorpus = readFileToCorpus('dummy.txt')
    print('movieCorpus:', movieCorpus)
    countWords(movieCorpus)
    movieVocab = getVocab(movieCorpus)
    movieIndex = createCorpusIndex(movieCorpus)
    printWordFrequencies(movieIndex, movieVocab)
    printCorpusConcordance("the", movieCorpus, movieIndex)

