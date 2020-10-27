# VizWiz with SAN
This repository contains a baseline attempt at using the SAN network used on VQA dataset and train it on the VizWiz dataset. The code based off aspects of the two following repositories:

[Visual Question Answering]: https://github.com/Shivanshu-Gupta/Visual-Question-Answering (SAN model)
[VizWiz]: https://github.com/DenisDsh/VizWiz-VQA-PyTorch (data processing)

The SAN model is mainly referenced from the first repository, and the data preprocessing is referenced from the VizWiz repository.

The dataset of VizWiz can be found from: https://vizwiz.org/tasks-and-datasets/vqa/
Please download training, validation, and annotation set.

Preprocessing is done to extract image embeddings to speed up training. The words are only parsed into a vocabulary library, and is not preprocessed into word embeddings beforehand. To preprocess the refer to the code within 'preprocessing' folder (referenced from VizWiz repository). It will store the image embeddings into a .h5 file that is indexed by the image id.

The processed image embedding will be size of (batch x 2048 x 14 x 14). In total with the original image set, it will take up at least 40 GB of space.

To run:

```sh
python ./preprocessing/create_vocabs.py

python ./preprocessing/image_features_extraction.py
```

Store all input data within a folder named prepro_data. It should contain: vocab.json, val.h5, train.h5, and a directory called Annotations (downloaded from VizWiz website).

The running configurations are stored at ./config/config_san_sgd_vizwiz.yml, please edit the file directors to corresponding location of where data are stored. The configuration is adapted based on combination of the config file from the two repostiories. If embeddings are given, then the images within the data section will not be used.

To run:
```sh
python main.py --config ./config/config_san_sgd_vizwiz.yml
```

The runtime per epoch is less than 1 minute per epoch, at batch size 32 on one GPU.

The training loss and acc will be stored in ./vqa/san_sgd_vizwiz_0.01
