# Deep Knowledge-Aware Network for News Recommendation

## Quick Start
To train model
```shell
python train.py
```

To evaluate model
```shell
python evaluation.py
```

## Data
Download GloVe pre-trained word embedding
```shell
mkdir data && cd data
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip
```

Download MIND dataset
```shell
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
unzip MINDlarge_train.zip -d train
unzip MINDlarge_dev.zip -d val
unzip MINDlarge_test.zip -d test
rm MINDlarge_*.zip
```

Preprocess data into appropriate format
```shell
cd ..
python data_preprocess.py
```

## Config
Config is in AIO style, to modify hyper-parameters
```shell
vim config.py
```
