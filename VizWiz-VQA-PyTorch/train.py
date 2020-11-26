import argparse
import os.path
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch.autograd import Variable
from tqdm import tqdm

import models
import utils
from datasets import vqa_dataset

from torch.utils.tensorboard import SummaryWriter

def load_weights(model, config):
	if config['use_pretrained'] is False:
		return
	else:
		print("--*--" * 8)
		print("Starting Model Initialization")
		print("--*--" * 8)
		print("Loading Model from %s" % config['pretrained_model_path'])
		log = torch.load(config['pretrained_model_path'])
		dict_weights = log['weights']
		if config['use_full_weights']:
			print("Loading the full model weights")
			model.load_state_dict(dict_weights)
		else:
			if config['transfer_image_feature_extractor']:
				print("--*--" * 8)
				print("Loading the Image Feature Extractor weights")
				image_feature_extractor_weights = {'.'.join(k.split('.')[1:]) : v for k,v in dict_weights.items() \
									 if 'image_feature_extractor' in k}
				model.image_feature_extractor.load_state_dict(image_feature_extractor_weights)
				if config['freeze_image_feature_extractor']:
					print("Freezing the Image Feature Extractor weights")
					for param in model.image_feature_extractor.parameters():
						param.requires_grad = False
				print("--*--" * 8)
			if config['transfer_attention_network']:
				print("--*--" * 8)
				print("Loading the Attention Network weights")
				attention_network_weights = {'.'.join(k.split('.')[1:]) : v for k,v in dict_weights.items() \
									 if 'attention' in k}
				model.attention.load_state_dict(attention_network_weights)
				if config['freeze_attention_network']:
					print("Freezing the Attention Network weights")
					for param in model.attention.parameters():
						param.requires_grad = False
				print("--*--" * 8)
			
		print("--*--" * 8)
		print("Model Initialization Complete")
		print("--*--" * 8)

def train(model, item, optimizer):
	model.train()
	log_softmax = nn.LogSoftmax(dim=1).cuda()

	# tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}

	v = item['visual']
	q = item['question']
	a = item['answer']
	q_length = item['q_length']

	v = Variable(v.cuda(async=True))
	q = Variable(q.cuda(async=True))
	a = Variable(a.cuda(async=True))
	q_length = Variable(q_length.cuda(async=True))

	out = model(v, q, q_length)

	# This is the Soft-loss described in https://arxiv.org/pdf/1708.00584.pdf

	nll = -log_softmax(out)

	loss = (nll * a / 10).sum(dim=1).mean()
	acc = utils.vqa_accuracy(out.data, a.data).cpu()

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	return acc, loss

def evaluate(model, loader, tracker, tb_logger, epoch, split):
	model.eval()
	# tracker_class, tracker_params = tracker.MeanMonitor, {}
	tracker_class = utils.AvgMonitor

	predictions = []
	samples_ids = []
	accuracies = []

	tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
	# loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
	# acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
	loss_tracker = tracker_class()
	acc_tracker = tracker_class()
	log_softmax = nn.LogSoftmax(dim=1).cuda()

	with torch.no_grad():
		for batch_idx, item in enumerate(tq):
			v = item['visual']
			q = item['question']
			a = item['answer']
			sample_id = item['sample_id']
			q_length = item['q_length']

			v = Variable(v.cuda(async=True))
			q = Variable(q.cuda(async=True))
			a = Variable(a.cuda(async=True))
			q_length = Variable(q_length.cuda(async=True))

			out = model(v, q, q_length)

			# This is the Soft-loss described in https://arxiv.org/pdf/1708.00584.pdf

			nll = -log_softmax(out)

			loss = (nll * a / 10).sum(dim=1).mean()
			acc = utils.vqa_accuracy(out.data, a.data).cpu()

			# save predictions of this batch
			_, answer = out.data.cpu().max(dim=1)

			predictions.append(answer.view(-1))
			accuracies.append(acc.view(-1))
			# Sample id is necessary to obtain the mapping sample-prediction
			samples_ids.append(sample_id.view(-1).clone())

			# loss_tracker.append(loss.item())
			# acc_tracker.append(acc.mean())
			loss_tracker.update(loss.item(), v.shape[0])
			acc_tracker.update(acc.mean(), v.shape[0])
			fmt = '{:.4f}'.format
			# tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
			tq.set_postfix(loss=fmt(loss_tracker.value), acc=fmt(acc_tracker.value))

		tb_logger.add_scalar('val/loss', loss_tracker.value, global_step = epoch)
		tb_logger.add_scalar('val/accuracy', 100.0 * acc_tracker.value, global_step = epoch)
		predictions = list(torch.cat(predictions, dim=0))
		accuracies = list(torch.cat(accuracies, dim=0))
		samples_ids = list(torch.cat(samples_ids, dim=0))

	eval_results = {
		'answers': predictions,
		'accuracies': accuracies,
		'samples_ids': samples_ids,
		# 'avg_accuracy': acc_tracker.mean.value,
		# 'avg_loss': loss_tracker.mean.value
		'avg_accuracy': acc_tracker.value,
		'avg_loss': loss_tracker.value
	}

	return eval_results


def main():
	# Load config yaml file
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_config', default='config/default.yaml', type=str,
						help='path to a yaml config file')
	args = parser.parse_args()

	if args.path_config is not None:
		with open(args.path_config, 'r') as handle:
			config = yaml.load(handle)

	# generate log directory
	dir_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	path_log_dir = os.path.join(config['logs']['dir_logs'], dir_name)

	if not os.path.exists(path_log_dir):
		os.makedirs(path_log_dir)

	if args.path_config is not None:
		# Dump the config file
		with open(os.path.join(path_log_dir, "config.yaml"), "w") as f:
			yaml.dump(config, f)

	print('Model logs will be saved in {}'.format(path_log_dir))

	tb_logger = SummaryWriter(log_dir=path_log_dir)

	cudnn.benchmark = True

	# Generate datasets and loaders
	train_loader = vqa_dataset.get_loader(config, split='train')
	val_loader = vqa_dataset.get_loader(config, split='val')
	print('Got Loader')
	# model = nn.DataParallel(models.Model(config, train_loader.dataset.num_tokens)).cuda()
	model = models.Model(config, train_loader.dataset.num_tokens).cuda()

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
								 config['training']['lr'])

	# Load model weights if necessary
	load_weights(model, config['model']['pretrained'])

	tracker = utils.Tracker()

	min_loss = 10
	max_accuracy = 0

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=config['training']['patience'])

	path_best_accuracy = os.path.join(path_log_dir, 'best_accuracy_log.pth')
	path_best_loss = os.path.join(path_log_dir, 'best_loss_log.pth')

	# Save the model before we train.
	path_init = os.path.join(path_log_dir, 'init_log.pth')
	log_data = {
		'epoch': -1,
		'config': config,
		'weights': model.state_dict(),
		'vocabs': train_loader.dataset.vocabs,
	}
	torch.save(log_data, path_init)  # save model
	split = config['training']['train_split']

	for i in range(config['training']['epochs']):
		tq = tqdm(train_loader, desc='{} E{:03d}'.format(split, i), ncols=0)

		for batch_idx, item in enumerate(tq):
			tracker_class = utils.AvgMonitor
			loss_tracker = tracker_class()
			acc_tracker = tracker_class()
			
			v = item['visual']
			
			acc, loss = train(model, item, optimizer)
			
			loss_tracker.update(loss.item(), v.shape[0])
			acc_tracker.update(acc.mean(), v.shape[0])
			tb_logger.add_scalar('train/loss', loss.item(), global_step = len(tq) * i + batch_idx)
			tb_logger.add_scalar('train/accuracy', 100.0 * acc.mean(), global_step = len(tq) * i + batch_idx)
			fmt = '{:.4f}'.format
			# tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
			tq.set_postfix(loss=fmt(loss_tracker.value), acc=fmt(acc_tracker.value))

			# If we are training on the train split (and not on train+val) we can evaluate on val
			if config['training']['train_split'] == 'train' and batch_idx != 0 and batch_idx % config['training']['val_freq'] == 0:
				eval_results = evaluate(model, val_loader, tracker, tb_logger, epoch=i, split='val')
				# Anneal LR and log it
				scheduler.step(eval_results['avg_accuracy'])
				tb_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step = i)
				# save all the information in the log file
				log_data = {
					'epoch': i,
					'config': config,
					'weights': model.state_dict(),
					'eval_results': eval_results,
					'vocabs': train_loader.dataset.vocabs,
				}

				# Save logs after every epoch
				epoch_log_path = os.path.join(path_log_dir, 'epoch_{}_log.pth'.format(i))
				torch.save(log_data, epoch_log_path)

				# save logs for min validation loss and max validation accuracy
				if eval_results['avg_loss'] < min_loss:
					torch.save(log_data, path_best_loss)  # save model
					min_loss = eval_results['avg_loss']  # update min loss value

				if eval_results['avg_accuracy'] > max_accuracy:
					torch.save(log_data, path_best_accuracy)  # save model
					max_accuracy = eval_results['avg_accuracy']  # update max accuracy value

	# # Save final model
	# log_data = {
	# 	'config': config,
	# 	'weights': model.state_dict(),
	# 	'vocabs': train_loader.dataset.vocabs,
	# }

	# path_final_log = os.path.join(path_log_dir, 'final_log.pth')
	# torch.save(log_data, path_final_log)


if __name__ == '__main__':
	main()
