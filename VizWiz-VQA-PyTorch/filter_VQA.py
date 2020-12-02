import os
import json
import nltk
import argparse
import numpy as np
from tqdm import tqdm
# from textblob import TextBlob

from preprocessing.preprocessing_utils import prepare_single_question
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')

THRESH = 1.0

def read_vocab(path):
	vocab = json.load(open(path, 'r'))
	return vocab['question'].keys()

def load_questions(path):
	questions_bank = json.load(open(path, 'r'))['questions']
	questions_dict = dict()
	for question in questions_bank:
		questions_dict[question['question_id']] = question['question']
	return questions_dict

def load_annotations(path):
	return json.load(open(path, 'r'))

def match_ratio(question, vocab):
	# Noun based filtering
	is_noun = lambda pos: pos[:2] == 'NN'
	nouns = [word for (word, pos) in nltk.pos_tag(question) if is_noun(pos)]
	if len(nouns) == 0:
		return 0.0
	else:
		matched_words = [x for x in nouns if x in vocab]
		matches = (1.0 * len(matched_words)/len(nouns))
	return matches

	# Stop words based matching
	# filtered_words = [x for x in question if x not in english_stop_words]
	# matched_words = [x for x in filtered_words if x in vocab]
	# if len(matched_words) == 0:
	# 	matches = -1.0
	# else:
	# 	matches = (1.0 * len(matched_words)/len(filtered_words))
	# return filtered_words, matched_words, matches

def clone_annotations(annotations):
	'''
	Clone the meta data and not the actual annotations
	'''
	new_annotations = {}
	new_annotations['info'] = annotations['info']
	new_annotations['license'] = annotations['license']
	new_annotations['data_type'] = annotations['data_type']
	new_annotations['data_subtype'] = annotations['data_subtype']
	new_annotations['annotations'] = []
	return new_annotations

def save_annotations(path, annotations):
	with open(path, 'w') as f:
		json.dump(annotations, f)

def run(args):
	vocab = read_vocab(args.vocab_path)
	questions = load_questions(args.input_questions_path)
	annotations = load_annotations(args.input_annotations_path)
	new_annotations = clone_annotations(annotations)
	matches = np.zeros(len(annotations['annotations']))
	for idx, ann in enumerate(tqdm(annotations['annotations'])):
		ques_id = ann['question_id']
		ques_string = questions[ques_id]
		ques_tokenized = prepare_single_question(ques_string)
		nps = match_ratio(ques_tokenized, vocab)
		matches[idx] = nps
		# Good annotations - Keep it!
		if nps >= THRESH:
			new_annotations['annotations'].append(ann)
	np.save('matches_NN.npy', matches)
	save_annotations(args.output_path, new_annotations)

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_questions_path', dest='input_questions_path', type=str, default='/home/ubuntu/data_vqa/Annotations/questions/train_questions.json')
	parser.add_argument('--input_annotations_path', dest='input_annotations_path', type=str, default='/home/ubuntu/data_vqa/Annotations/train.json')
	parser.add_argument('--output_path', dest='output_path', type=str, default='/home/ubuntu/data_vqa/Annotations/train_filtered.json')
	parser.add_argument('--vocab_path', dest='vocab_path', type=str, default='/home/ubuntu/prepro_data/viz_wiz/vocabs.json')

	args = parser.parse_args()
	run(args)