### Links

* https://www.kaggle.com/zhugds/resnet34-with-rgby-fast-ai-fork/notebook
* https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
* https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72895
* https://github.com/spytensor/kaggle_human_protein_baseline
* https://github.com/wdhorton/protein-atlas-fastai
* model distillation (e.g. nasnet & nasnet-mobile) -> https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/67604#413479
* f1_loss: https://www.kaggle.com/manyfoldcv/an-end-to-end-starter-kit-in-pytorch-xception/notebook
* https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/68678


### Ideas

* find and extract individual cells and classify those
* learn from training samples which have only 1 label
* use thresholds which achieve a similar per-class distribution of the different proteins as in the validation set
* normalize different channels separately
* any predictions with no label? if not, handle this when predicting
* split image e.g. into 2x2 subimages, train/predict independently, combine predictions
* make sure that the number of predictions is within [1,x], where x is the max number of targets in the train data
