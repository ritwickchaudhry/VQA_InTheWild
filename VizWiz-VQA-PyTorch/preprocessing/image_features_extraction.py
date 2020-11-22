# TODO : Generalize with device instead of gpu/cpu
import argparse
import time
import os
import h5py
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import yaml
from torch.autograd import Variable
from tqdm import tqdm

from images import ImageDataset, get_transform
import pdb
from pdb import set_trace as bp

class NetFeatureExtractor(nn.Module):

    def __init__(self):
        super(NetFeatureExtractor, self).__init__()
        self.model = models.resnet152(pretrained=True)
        # PyTorch models available in torch.utils.model_zoo  require an input of size 224x224.
        # This is because of the avgpooling that is fixed 7x7.
        # By using AdaptiveAvgPool2 we can feed to the network images with higher resolution (448x448)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

        # Save attention features (tensor)
        def save_att_features(module, input, output):
            self.att_feat = output

        # Save no-attention features (vector)
        def save_noatt_features(module, input, output):
            self.no_att_feat = output

        # This is a forward hook. Is executed each time forward is executed
        self.model.layer4.register_forward_hook(save_att_features)
        self.model.avgpool.register_forward_hook(save_noatt_features)

    def forward(self, x):
        self.model(x)
        return self.no_att_feat, self.att_feat  # [batch_size, 2048], [batch_size, 2048, 14, 14]

def main():
    # Load config yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='/home/ubuntu/VQA_InTheWild/VizWiz-VQA-PyTorch/config/default.yaml', type=str,
                        help='path to a yaml config file')
    args = parser.parse_args()

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.load(handle)
            # config = config['images']

    # Benchmark mode is good whenever your input sizes for your network do not vary
    cudnn.benchmark = True

    net = NetFeatureExtractor().cuda()
    net.eval()
    # Resize, Crop, Normalize
    transform = get_transform(config)
    dataset = ImageDataset(config['images']['dir'], transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['images']['preprocess_batch_size'],
        num_workers=config['images']['preprocess_data_workers'],
        shuffle=False,
        pin_memory=True,
    )

    h5_file = h5py.File(config['images']['path_features'], 'w')

    dummy_input = Variable(torch.ones(1, 3, config['images']['img_size'], config['images']['img_size']), volatile=True).cuda()
    _, dummy_output = net(dummy_input)

    att_features_shape = (
        len(data_loader.dataset),
        dummy_output.size(1),
        dummy_output.size(2),
        dummy_output.size(3)
    )

    noatt_features_shape = (
        len(data_loader.dataset),
        dummy_output.size(1),
        1,
        1
    )

    h5_att = h5_file.create_dataset('att', shape=att_features_shape, dtype='float16')
    h5_noatt = h5_file.create_dataset('noatt', shape=noatt_features_shape, dtype='float16')

    # save order of extraction
    dt = h5py.special_dtype(vlen=str)
    img_names = h5_file.create_dataset('img_name', shape=(len(data_loader.dataset),), dtype=dt)

    begin = time.time()
    end = time.time()

    print('Extracting features ...')
    idx = 0
    delta = config['preprocess_batch_size']

    for i, inputs in enumerate(tqdm(data_loader)):
        inputs_img = Variable(inputs['visual'].cuda(async=True), volatile=True)
        no_att_feat, att_feat = net(inputs_img)

        # bn, out_size, H, W = att_feat.shape
        # att_feat = att_feat.permute(0, 2, 3, 1)
        # att_feat_save = torch.reshape(att_feat, (bn, H*W, out_size))

        # for i in range(len(inputs['name'])):
        #     img_name = os.path.splitext(inputs['name'][i])[0]
        #     path = '{}.pth'.format(os.path.join(config['path_features'], img_name))
        #     torch.save(att_feat_save[i,:,:], path)

        # reshape (batch_size, 2048)
        # no_att_feat = no_att_feat.view(-1, 2048)
        h5_noatt[idx:idx + delta,:,:,:] = no_att_feat.data.cpu().numpy().astype('float16')
        h5_att[idx:idx + delta, :, :,:] = att_feat.data.cpu().numpy().astype('float16')
        img_names[idx:idx + delta] = inputs['name']

        idx += delta
    h5_file.close()

    end = time.time() - begin

    print('Finished in {}m and {}s'.format(int(end / 60), int(end % 60)))
    print('Created file : ' + config['path_features'])


if __name__ == '__main__':
    main()
