import numpy as np
import tensorflow as tf
import numpy as np
import os
from attenvis import AttentionVis
av = AttentionVis()

dirname = os.path.dirname(__file__)

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
HUNGARIAN_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 14

def pad_corpus(hungarian, english):
	HUNGARIAN_padded_sentences = []
	HUNGARIAN_sentence_lengths = []
	for line in hungarian:
		padded_hungarian = line[:HUNGARIAN_WINDOW_SIZE-1]

		padded_hungarian += [STOP_TOKEN] + [PAD_TOKEN] *  (HUNGARIAN_WINDOW_SIZE - len(padded_hungarian)-1)
		HUNGARIAN_padded_sentences.append(padded_hungarian)

	ENGLISH_padded_sentences = []
	ENGLISH_sentence_lengths = []
	for line in english:
		padded_ENGLISH = line[:ENGLISH_WINDOW_SIZE-1]
		padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN] + [PAD_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_ENGLISH)-1)
		ENGLISH_padded_sentences.append(padded_ENGLISH)

	return HUNGARIAN_padded_sentences, ENGLISH_padded_sentences

def build_vocab(sentences):
	"""
  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
  Convert sentences to indexed.

	:param vocab:  dictionary. Key: word --> Value: unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	toBeStacked =[]
	for sentence in sentences:
		tokenized=[]
		for word in sentence:
			if word in vocab:
				tokenized.append(vocab[word])
			else:
				tokenized.append(vocab[UNK_TOKEN])
		toBeStacked.append(tokenized)
	# return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])
	return np.stack(toBeStacked)

def read_data(referenceFile, file_lang1, file_lang2):
	"""
  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	lang1_sentences = open(file_lang1).readlines()
	lang2_sentences = open(file_lang2).readlines()
	lang1_data, lang2_data = [],[]
	with open(referenceFile, 'rt', encoding='latin') as data_file:
		for i in range(6001):
			line = data_file.readline()
			index_lang1 = int(line.split('\t')[-2].split()[0])-1
			index_lang2 = int(line.split('\t')[-1].split()[0])-1
			lang1_data.append(lang1_sentences[index_lang1].split())
			lang2_data.append(lang2_sentences[index_lang2].split())

	return lang1_data, lang2_data

@av.get_data_func
def get_data(referenceFile, lang1File, lang2File):
	en_train, hu_train = read_data(referenceFile, lang1File, lang2File)
	en_test = en_train[:len(en_train)//2]
	hu_test = hu_train[:len(hu_train)//2]
	en_train = en_train[len(en_train)//2:]
	hu_train = hu_train[len(hu_train)//2:]

	# pad training/testing data
	padded_hu_train, padded_en_train = pad_corpus(hu_train, en_train)
	padded_hu_test, padded_en_test = pad_corpus(hu_test, en_test)

	# build vocab for hungarian
	hu_vocab, hu_padding_index = build_vocab(padded_hu_train)
	en_vocab, en_padding_index = build_vocab(padded_en_train)

	print('finished building vocab')
	# convert training and testing english sentences to list of IDS
	train_english = convert_to_id(en_vocab, padded_en_train)
	test_english = convert_to_id(en_vocab, padded_en_test)
	print('finished covert to id')
	# convert training and testing hungarian sentences to list of IDS
	train_hungarian = convert_to_id(hu_vocab, padded_hu_train)
	test_hungarian = convert_to_id(hu_vocab, padded_hu_test)
	print('******* finished get_data')

	return train_english, test_english, train_hungarian, test_hungarian, en_vocab, hu_vocab



if __name__ =='__main__':
	get_data(
		os.path.join(dirname, '../data/references.txt'),
		os.path.join(dirname, '../data/en-data.txt'),
		os.path.join(dirname, '../data/hu-data.txt'))
	print("Preprocessing complete.")
