########################################
## CS447 Natural Language Processing  ##
##           Homework 1               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import random
import math
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
	""" Reads in the text file f which contains one sentence per line.
	"""
	if os.path.isfile(f):
		file = open(f, "r") # open the input file in read-only mode
		i = 0 # this is just a counter to keep track of the sentence numbers
		corpus = [] # this will become a list of sentences
		print("Reading file ", f)
		for line in file:
			i += 1
			sentence = line.split() # split the line into a list of words
			#append this lis as an element to the list of sentences
			corpus.append(sentence)
			if i % 1000 == 0:
		#print a status message: str(i) turns int i into a string
		#so we can concatenate it
				sys.stderr.write("Reading sentence " + str(i) + "\n")
		#endif
	#endfor
		return corpus
	else:
	#ideally we would throw an exception here, but this will suffice
		print("Error: corpus file ", f, " does not exist")
		sys.exit() # exit the script
	#endif
#enddef


# Preprocess the corpus to help avoid sess the corpus to help avoid sparsity
def preprocess(corpus):
	#find all the rare words
	freqDict = defaultdict(int)
	for sen in corpus:
		for word in sen:
		   freqDict[word] += 1
	#endfor
	#endfor

	#replace rare words with unk
	for sen in corpus:
		for i in range(0, len(sen)):
			word = sen[i]
			if freqDict[word] < 2:
				sen[i] = UNK
		#endif
	#endfor
	#endfor

	#bookend the sentences with start and end tokens
	for sen in corpus:
		sen.insert(0, start)
		sen.append(end)
	#endfor
	
	return corpus
#enddef

def preprocessTest(vocab, corpus):
	#replace test words that were unseen in the training with unk
	for sen in corpus:
		for i in range(0, len(sen)):
			word = sen[i]
			if word not in vocab:
				sen[i] = UNK
		#endif
	#endfor
	#endfor
	
	#bookend the sentences with start and end tokens
	for sen in corpus:
		sen.insert(0, start)
		sen.append(end)
	#endfor

	return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token
limit = 100 # longest length of generated sentence
#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
	# Initialize and train the model (ie, estimate the model's underlying probability
	# distribution from the training corpus)
	def __init__(self, corpus):
		print("""Your task is to implement five kinds of n-gram language models:
	  a) an (unsmoothed) unigram model (UnigramModel)
	  b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
	  c) an unsmoothed bigram model (BigramModel)
	  d) a bigram model smoothed using absolute discounting (SmoothedBigramModelAD)
	  e) a bigram model smoothed using kneser-ney smoothing (SmoothedBigramModelKN)
	  """)
	#enddef

	# Generate a sentence by drawing words according to the 
	# model's probability distribution
	# Note: think about how to set the length of the sentence 
	#in a principled way
	def generateSentence(self):
		print("Implement the generateSentence method in each subclass")
		return "mary had a little lamb ."
	#emddef

	# Given a sentence (sen), return the probability of 
	# that sentence under the model
	def getSentenceProbability(self, sen):
		print("Implement the getSentenceProbability method in each subclass")
		return 0.0
	#enddef

	# Given a corpus, calculate and return its perplexity 
	#(normalized inverse log probability)
	def getCorpusPerplexity(self, corpus):
		print("Implement the getCorpusPerplexity method")
		return 0.0
	#enddef

	# Given a file (filename) and the number of sentences, generate a list
	# of sentences and write each to file along with its model probability.
	# Note: you shouldn't need to change this method
	def generateSentencesToFile(self, numberOfSentences, filename):
		filePointer = open(filename, 'w+')
		for i in range(0,numberOfSentences):
			sen = self.generateSentence()
			prob = self.getSentenceProbability(sen)

			stringGenerated = str(prob) + " " + " ".join(sen) 
			print(stringGenerated, end="\n", file=filePointer)
			
	#endfor
	#enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
	def __init__(self, corpus):
		# stores the frequency of each word
		self.freq_dict = defaultdict(float)
		self.corpus_size = 0.0 # total number of word tokens in the corpus
		for sentence in corpus:
			for word in sentence:
				if word == start:
					continue
				if word in self.freq_dict:
					self.freq_dict[word] += 1
				else:
					self.freq_dict[word] = 1
				self.corpus_size += 1.0
		# number of word types in corpus
		self.num_word_type = len(self.freq_dict)

	# Returns the probability of word in the distribution
	def prob(self, word):
		# print("unsmoothed prob")
		return self.freq_dict[word]/self.corpus_size

	# Generate a single random word according to the distribution
	def draw(self):
		rand = random.random()
		for word in self.freq_dict.keys():
			rand -= self.prob(word)
			if rand <= 0.0:
				return word

	# Generate a sentence by drawing words according to the 
	# model's probability distribution
	# Note: think about how to set the length of the sentence 
	#in a principled way
	def generateSentence(self):
		word_list = [start] # list of generated word in sequence
		word = self.draw() # generate the first word
		length = 1
		while (word != end) and length < limit:
			word_list.append(word)
			word = self.draw() # generate the next word
			length += 1
		word_list.append(end)
		# print(word_list)
		return word_list

	# Given a sentence (a list of words including start and end), return the probability of 
	# that sentence under the model
	def getSentenceProbability(self, sen):
		# print('The sentence is:', sen)
		# new_sen = copy.deepcopy(sen[1:]) # exclude the start symbol
		sen.pop(0)
		# assert new_sen == sen
		# print('new_sen:', new_sen)
		prob_sen = 1.0
		# print(sen)
		for word in sen:
			prob_sen = prob_sen * self.prob(word)
		return prob_sen

	# Given a corpus, calculate and return its perplexity 
	#(normalized inverse log probability)
	def getCorpusPerplexity(self, corpus):
		prob_corpus = 0.0
		corpus_size = 0.0
		for sentence in corpus:
			# print(sentence)
			prob_corpus += math.log(self.getSentenceProbability(sentence))
			corpus_size += len(sentence) 
		return math.exp(-1.0 / corpus_size * prob_corpus)
   

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(UnigramModel):
	def __init__(self, corpus):
		UnigramModel.__init__(self, corpus)

	# Returns the probability of word in the distribution
	def prob(self, word):
		# print("smoothed prob")
		# print(word)
		if word in self.freq_dict:
			return (self.freq_dict[word] + 1) / (self.corpus_size + self.num_word_type) * 1.0
		else:
			return 1.0 / (self.corpus_size + self.num_word_type)

	def getSentenceProbability(self, sen):
		# print('The sentence is:', sen)
		# new_sen = copy.deepcopy(sen[1:]) # exclude the start symbol
		# sen.pop(0)
		# assert new_sen == sen
		# print('new_sen:', new_sen)
		prob_sen = 1.0
		# print(sen)
		for word in sen:
			prob_sen = prob_sen * self.prob(word)
		return prob_sen

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
	def __init__(self, corpus):
		# key: (previous_word, word); value: # of occurences of the tuple in corpus
		self.freq_dict = {} 
		# store the frequencies of all (word i-1, word i) in freq_dict 
		for sentence in corpus:
			previous_word = None
			for word in sentence:
				if previous_word != None:
					if (previous_word, word) in self.freq_dict:
						self.freq_dict[(previous_word, word)] += 1
					else:
						self.freq_dict[(previous_word, word)] = 1
				previous_word = word

	def generateSentence(self):
		word_list = [start]
		# initialze the first word as start symbol
		word = start
		length = 1
		while word != end and length < limit:
			next_word = self.gen_next_word(word)
			word_list.append(next_word)
			word = next_word
			length += 1
		word_list.append(end)
		return word_list

	# argument: current word
	# return value: next word generated
	def gen_next_word(self, word):
		# store all (key, val) with previous_word = word
		word_dict = {}
		total = 0.0
		for key, val in self.freq_dict.items():
			if key[0] == word:
				word_dict[key] = val
				total += val

		# generate next word
		rand = random.random()
		for word_pair, count in word_dict.items():
			rand -= count / total * 1.0
			if rand <= 0.0:
				return word_pair[1]

	# calculate the probability of (word1, word2) sequence in a dictionary 
	# word1: previous word, word2: current word
	def prob(self, dictionary, word1, word2):
		c1 = 0 # count(word1)
		c12 = 0 # count(word1, word2)
		for key, val in dictionary.items():
			if key[0] == word1:
				c1 += val
				if key[1] == word2:
					c12 += val
		return c12 / c1 * 1.0


	def getSentenceProbability(self, sen):
		sen.insert(0, start)
		# if sentence contains 0 or 1 word...
		if len(sen) < 1:
			return 0
		prob = 1.0
		# iterate each pair of (word1, word2), multiply prob with p(word1, word2)
		for i in range(0, len(sen)-1):
			if (sen[i], sen[i+1]) in self.freq_dict:
				prob *= self.prob(self.freq_dict, sen[i], sen[i+1])
			else:
				return 0
		return prob

	def getCorpusPerplexity(self, corpus):
		prob_corpus = 0.0
		corpus_size = 0.0

		for sentence in corpus:
			# print(sentence)
			prob_corpus += math.log(self.getSentenceProbability(sentence))
			corpus_size += len(sentence) - 1
		return math.exp(-1.0 / corpus_size * prob_corpus)


# Smoothed bigram language model (use absolute discounting for smoothing)
class SmoothedBigramModelAD(LanguageModel):
	# already have: unigram freq_dict, corpus_size, num_word_type
	def __init__(self, corpus):
		# key: (previous_word, word); value: # of occurences of the tuple in corpus
		self.freq_dict = defaultdict(float)
		self.freq_dict_pair = defaultdict(float)
		self.corpus_size = 0.0
		# store the frequencies of all (word i-1, word i) in freq_dict 
		for sentence in corpus:
			previous_word = None
			for word in sentence:
				self.freq_dict[word] += 1.0
				self.corpus_size += 1
				if previous_word != None:
					self.freq_dict_pair[(previous_word, word)] += 1.0
				previous_word = word

		# find D
		n1 = 0
		n2 = 0
		for key, val in self.freq_dict_pair.items():
			if val == 1:
				n1 += 1
			elif val == 2:
				n2 += 1
		self.D = n1 * 1.0 / (n1 + 2 * n2)

		# intialize a dictionary for all S(w) for future use
		self.S = {}



	def generateSentence(self):
		word_list = [start]
		# initialze the first word as start symbol
		word = start
		length = 1
		while word != end and length < limit:
			# print(word)
			next_word = self.gen_next_word(word)
			word_list.append(next_word)
			word = next_word
			length += 1
		# print(word_list)
		word_list.append(end)
		return word_list

	# argument: current word
	# return value: next word generated
	def gen_next_word(self, word):
		# store all (key, val) with previous_word = word
		word_dict = {}
		total = 0.0
		for key, val in self.freq_dict_pair.items():
			if key[0] == word:
				word_dict[key] = val
				total += val
		# print(word)
		# print('word dict:', word_dict)
		# generate next word
		rand = random.random()
		for word_pair, count in word_dict.items():
			rand -= self.pair_prob(word_pair[0], word_pair[1])
			if rand <= 0.0:
				return word_pair[1]

	# find the number of word types that occurs immediately after word
	def findS(self, word):
		# print('word:', word)
		# if word already in self.S
		if word in self.S:
			return self.S[word]
		# if not, calculate and update self.S[word]
		s = 0
		next_word_list = []
		for key in self.freq_dict_pair:
			if key[0] == word and key[1] not in next_word_list and self.freq_dict_pair[key] > 0:
				s += 1
				next_word_list.append(key[1])
		self.S[word] = s # update self.S
		return s
	

	def prob(self, word):
		return (self.freq_dict[word] + 1) / (self.corpus_size + len(self.freq_dict) - self.freq_dict[start] - 1) * 1.0

	# calculate the probability of (word1, word2) sequence in a dictionary 
	# word1: previous word, word2: current word
	def pair_prob(self, word1, word2):
		c1 = self.freq_dict[word1]
		c12 = self.freq_dict_pair[(word1, word2)]
		# inherent from SmoothedUnigram
		pL2 = self.prob(word2)
		
		s = self.findS(word1)
		pAD = max(c12 - self.D, 0) / c1 * 1.0 + self.D * s * pL2 / c1 * 1.0
		# print('pAD:', pAD)
		# use log probability
		return math.log(pAD)


	def getSentenceProbability(self, sen):
		# if sentence contains 0 or 1 word...
		if len(sen) < 1:
			return 0
		# use log probability
		prob = 0.0
		# iterate each pair of (word1, word2), multiply prob with p(word1, word2)
		for i in range(0, len(sen)-1):
			temp = self.pair_prob(sen[i], sen[i+1])
			prob = prob + temp
			# print(temp)
			# prob = prob + self.pair_prob(self.freq_dict_pair, sen[i], sen[i+1])

		return math.exp(prob)

	def getCorpusPerplexity(self, corpus):
		prob_corpus = 0.0 # use log probability
		N = 0.0
		for sentence in corpus:
			prob_corpus += math.log(self.getSentenceProbability(sentence))
			N += len(sentence) - 1

		return math.exp(-1.0 / N * prob_corpus)


# Smoothed bigram language model (use absolute discounting and kneser-ney for smoothing)
class SmoothedBigramModelKN(SmoothedBigramModelAD):
	def __init__(self, corpus):
		SmoothedBigramModelAD.__init__(self, corpus)
		self.num_word_type = len(self.freq_dict_pair)
		self.PC = {}

	# find |w: c(w, w') > 0|
	def find_PC(self, word):
		# find numerator (same logic as self.S)
		if word in self.PC:
			return self.PC[word]

		numerator = 0.0
		for key in self.freq_dict_pair:
			if key[1] == word and self.freq_dict_pair[key] > 0:
				numerator += 1
		self.PC[word] = numerator / self.num_word_type * 1.0
		return self.PC[word]


	# calculate the probability of (word1, word2) sequence in a dictionary 
	# word1: previous word, word2: current word
	def pair_prob(self, word1, word2):
		c1 = self.freq_dict[word1]
		c12 = self.freq_dict_pair[(word1, word2)]
		# find P_C(word2)
		pc2 = self.find_PC(word2)
		s = self.findS(word1)
		pAD = max(c12 - self.D, 0) / c1 * 1.0 + self.D * s * pc2 / c1 * 1.0
		return math.log(pAD)

# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
	def __init__(self, corpus):
		self.counts = defaultdict(float)
		self.total = 0.0
		self.train(corpus)
	#endddef

	# Add observed counts from corpus to the distribution
	def train(self, corpus):
		for sen in corpus:
			for word in sen:
				if word == start:
					continue
				self.counts[word] += 1.0
				self.total += 1.0
			#endfor
		#endfor
	#enddef

	# Returns the probability of word in the distribution
	def prob(self, word):
		return self.counts[word]/self.total
	#enddef

	# Generate a single random word according to the distribution
	def draw(self):
		rand = random.random()
		for word in self.counts.keys():
			rand -= self.prob(word)
			if rand <= 0.0:
				return word
		#endif
	#endfor
	#enddef
#endclass

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
	#read your corpora
	trainCorpus = readFileToCorpus('train.txt')
	trainCorpus = preprocess(trainCorpus)
	
	posTestCorpus = readFileToCorpus('pos_test.txt')
	negTestCorpus = readFileToCorpus('neg_test.txt')
	
	vocab = set()
	# Please write the code to create the vocab over here before the function preprocessTest
	print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")


	posTestCorpus = preprocessTest(vocab, posTestCorpus)
	negTestCorpus = preprocessTest(vocab, negTestCorpus)
	'''
	# Run sample unigram dist code
	unigramDist = UnigramDist(trainCorpus)
	print("Sample UnigramDist output:")
	print("Probability of \"vader\": ", unigramDist.prob("vader"))
	print("Probability of \""+UNK+"\": ", unigramDist.prob(UNK))
	print("\"Random\" draw: ", unigramDist.draw())
	'''
	unsmoothed_unigram = UnigramModel(trainCorpus)
	unsmoothed_unigram.generateSentencesToFile(20, "unigram_output.txt")

	smoothed_unigram = SmoothedUnigramModel(trainCorpus)
	smoothed_unigram.generateSentencesToFile(20, "smooth_unigram_output.txt")

	unsmoothed_bigram = BigramModel(trainCorpus)
	unsmoothed_bigram.generateSentencesToFile(20, "bigram_output.txt")

	smoothed_bigram_ad = SmoothedBigramModelAD(trainCorpus)
	smoothed_bigram_ad.generateSentencesToFile(20, "smooth_bigram_ad_output.txt")

	smoothed_bigram_kn = SmoothedBigramModelKN(trainCorpus)
	smoothed_bigram_kn.generateSentencesToFile(20, "smooth_bigram_kn_output.txt")