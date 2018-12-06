### Links

* https://www.kaggle.com/zhugds/resnet34-with-rgby-fast-ai-fork/notebook
* https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
* https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72895
* https://github.com/spytensor/kaggle_human_protein_baseline
* https://github.com/wdhorton/protein-atlas-fastai
* model distillation (e.g. nasnet & nasnet-mobile) -> https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/67604#413479
* f1_loss: https://www.kaggle.com/manyfoldcv/an-end-to-end-starter-kit-in-pytorch-xception/notebook
* https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/68678
* https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/73395
* https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/73410
* https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/73199
* focal loss: https://github.com/Prakashvanapalli/av_mckinesy_recommendation_challenge/blob/master/func.py#L55
* https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72534


### Ideas

* find and extract individual cells and classify those
* learn from training samples which have only 1 label
* use thresholds which achieve a similar per-class distribution of the different proteins as in the validation set
* normalize different channels separately
* any predictions with no label? if not, handle this when predicting
* split image e.g. into 2x2 subimages, train/predict independently, combine predictions
* make sure that the number of predictions is within [1,x], where x is the max number of targets in the train data
* use sklearn's f1_score
* similar images between train and test set? (https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72534)
* data leak: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/73395
* find correlation between different classes for multi-class targets in training data
* add lr finder mode to lr scheduling
* save the val/test predictions
* treat problem as image segmentation with green channel as mask


### Challenges

* class imbalance
  * visualize/explore
  * bce loss weights
  * focal loss
  * stratified train/test split (for multi target samples, take the most rare category for stratification)
  * stratified mini-batch sampling
  * do oversampling of classes with low frequencies
* overfitting
  * add dropout
  * apply data augmentation
  * use leaky ReLU
* thresholding
  * use per-class threshold which optimizes for that class only
  * use per-class threshold which replicates the distribution of that class (and still improves global score)
* reliance on f1score
  * fixed threshold of 0.5 might be misleading
  * use loss metric instead
* score
  * use stronger model
  * center loss for better discrimination
  * attention
  * use f1 score based loss
  * use f1 score based loss with weights
* prediction performance (special prize)
  * knowledge distillation
