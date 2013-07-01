ILC_LandmarkDetector
====================

facial landmarks detector training (MATLAB)

This project is used to train landmark detectors using linear SVM and logistic regression. The final outcome is the batch of linear classifiers for 88 landmarks of human faces.

Given training samples and landmark annotations, features are extracted in PrepareTrainingData.m. Linear SVMs are trained in SVM_Train.m. Finally, logistic.m performs logistic regression on results of linear SVMs. LandmarkDetectorTrain.m combines all the procedures into a whole.
init_train_script.m is the Matlab script file managing data I/O and alogrithm. Incoporate your own data by modifying this file.

Copyright (C) 2013 Feng Shuang

If you have any problems, please contact me via safeng@umich.edu.
