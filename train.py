import shutil
import time
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import pdb
from pdb import set_trace as bp
from scheduler import CustomReduceLROnPlateau
import json
from tqdm import tqdm
import csv

def train(epoch, model, dataloader, criterion, optimizer, use_gpu=False):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0.0
    example_count = 0.0
    step = 0
    # Pdb().set_trace()
    # Iterate over data.
    with tqdm(total=len(dataloader),desc='Train Epoch#{}'.format(epoch + 1)) as t:
        for item in dataloader:
            questions, images, image_ids, answers = item['question'], item['visual'], item['sample_id'], item['answer']
            # print('questions size: ', questions.size())
            if use_gpu:
                questions, images, image_ids, answers = questions.cuda(), images.cuda(), image_ids.cuda(), answers.cuda()
            questions, images, answers = Variable(questions).transpose(0, 1), Variable(images), Variable(answers)

            # zero grad
            optimizer.zero_grad()
            ans_scores = model(images, questions, image_ids)

            preds = torch.argmax(ans_scores, axis=1)
            ans_scores = ans_scores.float()
            answers = torch.tensor(answers, dtype=torch.int64).cuda().argmax(axis=1)
            loss = criterion(ans_scores, answers)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data
            running_corrects += torch.sum((preds == answers).data)
            example_count += answers.size(0)
            step += 1
            # if step % 5000 == 0:
            #     print('running loss: {}, running_corrects: {}, example_count: {}, acc: {}'.format(
            #         running_loss / example_count, running_corrects, example_count, (float(running_corrects) / example_count) * 100))
            # if step * batch_size == 40000:
            #     break
            t.set_postfix({'running loss': (running_loss / example_count).cpu().data, 
                          'running_corrects': (running_corrects).cpu().data, 
                          'example_count': example_count, 
                          'acc': (float(running_corrects) / example_count)})
            t.update(1) 

    loss = running_loss / example_count
    acc = (running_corrects / len(dataloader.dataset)) * 100
    print('Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                           acc, running_corrects, example_count))
    return loss, acc


def validate(model, dataloader, criterion, use_gpu=False):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0.0
    example_count = 0.0
    # Iterate over data.
    for item in dataloader:
        questions, images, image_ids, answers = item['question'], item['visual'], item['sample_id'], item['answer']
        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(
            ), images.cuda(), image_ids.cuda(), answers.cuda()
        questions, images, answers = Variable(questions).transpose(
            0, 1), Variable(images), Variable(answers)

        # zero grad
        ans_scores = model(images, questions, image_ids)
        preds = torch.argmax(ans_scores, axis=1) 
        ans_scores = ans_scores.float()
        answers = torch.tensor(answers, dtype=torch.int64).cuda().argmax(axis=1)
        loss = criterion(ans_scores, answers)

        # statistics
        running_loss += loss.data
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)
    loss = running_loss / example_count
    # acc = (running_corrects / example_count) * 100
    acc = (running_corrects / len(dataloader.dataset)) * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                                acc, running_corrects, example_count))
    return loss, acc


def train_model(model, data_loaders, criterion, optimizer, scheduler, save_dir, num_epochs=25, use_gpu=False, best_accuracy=0, start_epoch=0):
    print('Training Model with use_gpu={}...'.format(use_gpu))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = best_accuracy

    with open(save_dir + '/train_data.csv', 'a+', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        writer = csv.writer(csvfile)

        for epoch in range(start_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            train_begin = time.time()
            train_loss, train_acc = train(epoch,
                model, data_loaders['train'], criterion, optimizer, use_gpu)
            train_time = time.time() - train_begin
            print('Epoch Train Time: {:.0f}m {:.0f}s'.format(
                train_time // 60, train_time % 60))

            validation_begin = time.time()
            val_loss, val_acc = validate(
                model, data_loaders['val'], criterion, use_gpu)
            validation_time = time.time() - validation_begin
            print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(
                validation_time // 60, validation_time % 60))
            writer.writerow([epoch, train_loss.data.cpu(), train_acc.data.cpu(), val_loss.data.cpu(), val_acc.data.cpu()])

            # deep copy the model
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                best_model_wts = model.state_dict()

            save_checkpoint(save_dir, {
                'epoch': epoch,
                'best_acc': best_acc,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
            }, is_best)

            valid_error = 1.0 - val_acc / 100.0
            if type(scheduler) == CustomReduceLROnPlateau:
                scheduler.step(valid_error, epoch=epoch)
                if scheduler.shouldStopTraining():
                    print("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(
                        scheduler.unconstrainedBadEpochs, scheduler.maxPatienceToStopTraining))
                    # Pdb().set_trace()
                    break
            else:
                scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json(save_dir + "/all_scalars.json")
    writer.close()

    return model


def save_checkpoint(save_dir, state, is_best):
    savepath = save_dir + '/' + 'checkpoint.pth.tar'
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, save_dir + '/' + 'model_best.pth.tar')


def test_model(model, dataloader, itoa, outputfile, use_gpu=False):
    model.eval()  # Set model to evaluate mode
    example_count = 0
    test_begin = time.time()
    outputs = []

    # Iterate over data.
    for questions, images, image_ids, answers, ques_ids in dataloader:

        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(
            ), images.cuda(), image_ids.cuda(), answers.cuda()
        questions, images, answers = Variable(questions).transpose(
            0, 1), Variable(images), Variable(answers)
        # zero grad
        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)

        outputs.extend([{'question_id': ques_ids[i], 'answer': itoa[str(
            preds.data[i])]} for i in range(ques_ids.size(0))])

        if example_count % 100 == 0:
            print('(Example Count: {})'.format(example_count))
        # statistics
        example_count += answers.size(0)

    json.dump(outputs, open(outputfile, 'w'))
    print('(Example Count: {})'.format(example_count))
    test_time = time.time() - test_begin
    print('Test Time: {:.0f}m {:.0f}s'.format(test_time // 60, test_time % 60))
