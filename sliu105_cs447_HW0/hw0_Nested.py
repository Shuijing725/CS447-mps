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

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
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
			sentence = line.split() # split the line into a list of words
			corpus.append(sentence) # append this list as an element to the list of sentences
			if i % 1000 == 0:
				sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it
		return corpus
	else:
		print("Error: corpus file ", f, " does not exist")  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
		sys.exit() # exit the script

#-------------------------------------------
# Data output
#-------------------------------------------


# Print out corpus statistics:
# - how many sentences?
# - how many word tokens?
def printStats(corpus):
	print('Number of sentences:', len(corpus))
	word = 0
	for sentence in corpus:
		word += len(sentence)
	print('Number of words:', word)
		
#
#
#
def getVocab(corpus):
	vocab = []
	for sentence in corpus:
		for word in sentence:
			vocab.append(word)
	return sorted(set(vocab))

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
def printConcordance(sentence, word_i):
	""" print out the five words preceding word,
		the word at position i and the folllowing five words."""
	if word_i < len(sentence):
		start = max(word_i-5, 0)
		end = min(word_i+6, len(sentence))
		left = ' '.join(sentence[start:word_i])
		right = ' '.join(sentence[word_i+1:end])
		print(left.rjust(40), sentence[word_i].center(10), right.ljust(30))



#--------------------------------------------------------------
# Corpus analysis (tokens as class)
#--------------------------------------------------------------

# We use the class Token to point to individual tokens (words) in the corpus.
class Token:
	def __init__(self, s, w): # we need to initialize each instance of Token:
		self.sentence = s # sentence is the index of the sentence (in the corpus)
		self.word = w # word is the index of the word (in the sentence)

#--------------------------------------------------------------
# Corpus analysis (tokens as tuple (i, j))
#--------------------------------------------------------------

#
# Create an index that maps each word to all its positions in the corpus
# (tokens are encoded as a tuple)
#
def createCorpusIndex_TupleVersion(corpus):
	index = {}
	# i: index of a sentence in a corpus
	# j: index of a word in a sentence
	for i in range(len(corpus)):
		for j in range(len(corpus[i])):
			if corpus[i][j] in index:
				index[corpus[i][j]] += [(i, j)]
			else:
				index[corpus[i][j]] = [(i, j)]

	return index

# index: return value of the funtion above
def printWordFrequencies_TupleVersion(index, vocab):
	# create a dictionary: key = words in vocabulary, item = frequency of the word
	freq_dict = {}
	for key, item in index.items():
		freq_dict[key] = len(item)
	# sort the dictionary by frequency
	freq_dict_sorted = sorted(freq_dict.items(), key = itemgetter(1), reverse = True)
	
	for word in freq_dict_sorted:
		print("Word: ", word[0], ", frequency: ", word[1])

#
# Print out all occurrences of the specified word in the corpus indexed by index
# (tokens are encoded as a tuple)
# 
# index is the return value of function createCorpusIndex_TupleVersion
def printCorpusConcordance_TupleVersion(word, corpus, index):
	if word not in index:
		raise ValueError('input word not found in index')
		return

	for pos in index[word]:
		printConcordance(corpus[pos[0]], pos[1])

#
# Create an index that maps each word to all its positions in the corpus
# (tokens are encoded as a tuple)
#
def createCorpusIndex_ClassVersion(corpus):
	index = {}
	# i: index of a sentence in a corpus
	# j: index of a word in a sentence
	for i in range(len(corpus)):
		for j in range(len(corpus[i])):
			if corpus[i][j] in index:
				index[corpus[i][j]] += [Token(i, j)]
			else:
				index[corpus[i][j]] = [Token(i, j)]

	return index
#
# Prints out all words sorted by their frequency, in descending order
#
def printWordFrequencies_ClassVersion(index, vocab):
	# create a dictionary: key = words in vocabulary, item = frequency of the word
	freq_dict = {}
	for key, item in index.items():
		freq_dict[key] = len(item)
	# sort the dictionary by frequency
	freq_dict_sorted = sorted(freq_dict.items(), key = itemgetter(1), reverse = True)
	
	for word in freq_dict_sorted:
		print("Word: ", word[0], ", frequency: ", word[1])

#
# Print out all occurrences of the word 'word' in the corpus indexed by index 
# (tokens are encoded as a class)
#
def printCorpusConcordance_ClassVersion(word, corpus, index):
	if word not in index:
		raise ValueError('input word not found in index')
		return

	for token in index[word]:
		printConcordance(corpus[token.sentence], token.word)
	
#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
	movieCorpus = readFileToCorpus('movies.txt')
	printStats(movieCorpus)
	movieVocab = getVocab(movieCorpus)
	movieIndexTuples = createCorpusIndex_TupleVersion(movieCorpus)
	printWordFrequencies_TupleVersion(movieIndexTuples, movieVocab)
	printCorpusConcordance_TupleVersion("the", movieCorpus, movieIndexTuples)
	movieIndexClass = createCorpusIndex_ClassVersion(movieCorpus)
	printWordFrequencies_ClassVersion(movieIndexTuples, movieVocab)
	printCorpusConcordance_ClassVersion("the", movieCorpus, movieIndexClass)
