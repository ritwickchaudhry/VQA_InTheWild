import json
import os
import os.path

import h5py
import torch
import torch.utils.data as data
from pdb import set_trace as bp
from datasets.features import FeaturesDataset
from preprocessing.preprocessing_utils import prepare_questions, prepare_answers, encode_question, encode_answers, prepare_questions_pretrain
from preprocessing.images import ImageDataset, get_transform

def get_loader(config, split):
    """ Returns the data loader of the specified dataset split """
    if config['annotations']['type'] == 'vqa':
        split = VQA_Pretrain_Dataset(
                config,
                split
            )
    else:
        split = VQADataset(
            config,
            split
        )

    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config['training']['batch_size'],
        shuffle=True if split == 'train' or split == 'trainval' else False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config['training']['data_workers'],
        collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # Sort samples in the batch based on the question lengths in descending order.
    # This allows to pack the pack_padded_sequence when encoding questions using RNN
    batch.sort(key=lambda x: x['q_length'], reverse=True)
    return data.dataloader.default_collate(batch)

class VQA_Pretrain_Dataset(data.Dataset):
    def __init__(self, config, split):
        super(VQA_Pretrain_Dataset, self).__init__()

        with open(config['annotations']['path_vocabs'], 'r') as fd:
            vocabs = json.load(fd)

        annotations_dir = config['annotations']['dir']

        path_ann = os.path.join(annotations_dir, split + ".json")
        with open(path_ann, 'r') as fd:
            self.annotations = json.load(fd)['annotations']
        
        self.max_question_length = config['annotations']['max_length']
        self.split = split

        # vocab
        with open(config['annotations']['path_vocabs'], 'r') as fd:
            vocabs = json.load(fd)
        self.vocabs = vocabs
        self.token_to_index = self.vocabs['question']
        self.answer_to_index = self.vocabs['answer']

        # questions
        question_dir = config['annotations']['questions_dir']
        path_ques = os.path.join(question_dir, split + "_questions.json")
        with open(path_ques, 'r') as fd:
            self.questions_bank = json.load(fd)

        self.questions = prepare_questions_pretrain(self.annotations, self.questions_bank)
        self.questions = [encode_question(q, self.token_to_index, self.max_question_length) for q in
                          self.questions]  # encode questions and return question and question lenght
        # answers
        if self.split != 'test':
            self.answers = prepare_answers(self.annotations)
            self.answers = [encode_answers(a, self.answer_to_index) for a in
                            self.answers]  # create a sparse vector of len(self.answer_to_index) for each question containing the occurances of each answer
        
        if self.split == "train" or self.split == "trainval":
            self._filter_unanswerable_samples()

        # images
        self.name_to_id = dict()
        self.preprocessed = config['images']['preprocessed']
        if not self.preprocessed:
            transform = get_transform(config)
            img_names = ImageDataset(os.path.join(config['images']['dir'], split), transform=transform)
            self.name_to_id = {name: i for i, name in enumerate(img_names.get_image_names)}            
            self.features = img_names
        else:
            # load image names in feature extraction order
            feat_path = config['images']['path_features'] + split + '.h5'
            self.features = FeaturesDataset(feat_path, config['images']['mode'])
            with h5py.File(feat_path, 'r') as f:
                img_names = f['img_name'][()]
            self.name_to_id = {name: i for i, name in enumerate()}

        # names in the annotations, will be used to get items from the dataset
        self.img_names = [config['images']['dir_prefix']+'{:06d}'.format(s['image_id'])+'.jpg' for s in self.annotations]

    def _filter_unanswerable_samples(self):
        """
        Filter during training the samples that do not have at least one answer
        """
        a = []
        q = []
        annotations = []
        for i in range(len(self.answers)):
            if len(self.answers[i].nonzero()) > 0:
                a.append(self.answers[i])
                q.append(self.questions[i])

                annotations.append(self.annotations[i])
        self.answers = a
        self.questions = q
        self.annotations = annotations

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def __getitem__(self, i):

        item = {}
        item['question'], item['q_length'] = self.questions[i]
        if self.split != 'test':
            item['answer'] = self.answers[i]

        img_name = self.img_names[i]
        item['img_name'] = img_name
        feature_id = self.name_to_id[img_name]
        item['visual'] = self.features[feature_id] if self.preprocessed else self.features[feature_id]['visual']
        # collate_fn sorts the samples in order to be possible to pack them later in the model.
        # the sample_id is returned so that the original order can be restored during when evaluating the predictions
        item['sample_id'] = i

        return item

    def __len__(self):
        return len(self.questions)

class VQADataset(data.Dataset):
    """ VQA dataset, open-ended """

    def __init__(self, config, split):
        super(VQADataset, self).__init__()


        annotations_dir = config['annotations']['dir']

        path_ann = os.path.join(annotations_dir, split + ".json")
        with open(path_ann, 'r') as fd:
            self.annotations = json.load(fd)

        self.max_question_length = config['annotations']['max_length']
        self.split = split

        # vocab
        with open(config['annotations']['path_vocabs'], 'r') as fd:
            vocabs = json.load(fd)
        self.vocabs = vocabs
        self.token_to_index = self.vocabs['question']
        self.answer_to_index = self.vocabs['answer']

        # pre-process questions and answers
        self.questions = prepare_questions(self.annotations)
        self.questions = [encode_question(q, self.token_to_index, self.max_question_length) for q in
                          self.questions]  # encode questions and return question and question lenght

        if self.split != 'test':
            self.answers = prepare_answers(self.annotations)
            self.answers = [encode_answers(a, self.answer_to_index) for a in
                            self.answers]  # create a sparse vector of len(self.answer_to_index) for each question containing the occurances of each answer

        if self.split == "train" or self.split == "trainval":
            self._filter_unanswerable_samples()

        self.preprocessed = config['images']['preprocessed']
        if not self.preprocessed:
            transform = get_transform(config)
            img_names = ImageDataset(os.path.join(config['images']['dir'], split), transform=transform) # os.path.join(config['images']['dir'], split)
            self.name_to_id = {name: i for i, name in enumerate(img_names.get_image_names)}
            self.features = img_names
        else:
            # load image names in feature extraction order
            feat_path = config['images']['path_features'] + split + '.h5'
            self.features = FeaturesDataset(feat_path, config['images']['mode'])
            with h5py.File(feat_path, 'r') as f:
                img_names = f['img_name'][()]
            self.name_to_id = {name: i for i, name in enumerate()}

        # names in the annotations, will be used to get items from the dataset
        self.img_names = [s['image'] for s in self.annotations]

    def _filter_unanswerable_samples(self):
        """
        Filter during training the samples that do not have at least one answer
        """
        a = []
        q = []
        annotations = []
        for i in range(len(self.answers)):
            if len(self.answers[i].nonzero()) > 0:
                a.append(self.answers[i])
                q.append(self.questions[i])

                annotations.append(self.annotations[i])
        self.answers = a
        self.questions = q
        self.annotations = annotations

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def __getitem__(self, i):

        item = {}
        item['question'], item['q_length'] = self.questions[i]
        if self.split != 'test':
            item['answer'] = self.answers[i]

        img_name = self.img_names[i]
        item['img_name'] = img_name
        feature_id = self.name_to_id[img_name]
        item['visual'] = self.features[feature_id] if self.preprocessed else self.features[feature_id]['visual']
        # collate_fn sorts the samples in order to be possible to pack them later in the model.
        # the sample_id is returned so that the original order can be restored during when evaluating the predictions
        item['sample_id'] = i

        return item

    def __len__(self):
        return len(self.questions)