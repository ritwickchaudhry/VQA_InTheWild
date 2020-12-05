import os
import argparse
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def run(args):
	log_dir = args.path
	files = os.listdir(log_dir)
	matching_files = [x for x in files if x.startswith('event')]
	assert len(matching_files) == 1, "No or multiple tensorboard log files found"
	print(os.path.join(log_dir, matching_files[0]))
	ea = event_accumulator.EventAccumulator(
		os.path.join(log_dir, matching_files[0]),
	)
	ea.Reload()
	accuracies = np.array([x[2] for x in ea.Scalars('val/accuracy')])
	assert len(accuracies) >= 5, "Number of steps less than 5"
	return np.sort(accuracies)[-5:]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', dest='path', type=str, required=True)
	args = parser.parse_args()

	accuracies = run(args)
	print("Accuracies: {}, 5-Average: {}".format(accuracies, np.mean(accuracies)))