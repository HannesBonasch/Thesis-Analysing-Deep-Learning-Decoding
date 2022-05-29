# Analysing Deep Learning Decoding Methods on Multiple ERP Paradigms

This is the repository for my master thesis called "Analysing Deep Learning Decoding Methods on Multiple ERP Paradigms".

The following is a very short, mostly visual look into the thesis, for more detail check out the pdf and the notebooks.

## Abstract:

Deep learning methods have successfully advanced many fields of research with their
ability to learn complex features from data. 

While they have been used successfully in
BCI research, their use for cognitive science, where the increased complexity of deep
learning methods could reveal novel insights about how our brain functions, is just
starting to be explored. 

In this thesis, we look at three established EEG decoding models,
EEGNet, Shallow ConvNet, and Deep ConvNet, on the ERP CORE dataset, which includes
seven different ERP components. 

We will look at how parameters like model architecture,
sample count, and preprocessing affect decoding accuracies, compare subject accuracies
across the different ERP paradigms, and look at how feature attribution can be used
to explain the decisions of our networks, as well as gain new insights into cognitive
processes. 

We conclude that that deep learning can be a valuable tool for cognitive
science that needs further research to reach its full potential.

## Dataset:

We decided to use the ERP CORE dataset (https://erpinfo.org/erp-core), as it includes seven different ERP paradigms.

Each ERP paradigm consists of ten minutes of raw EEG data where the forty subjects are shown different types of stimuli.

This is what raw EEG data looks like in MNE:

![raw_eeg](https://user-images.githubusercontent.com/28629116/170886227-24b51ffb-2873-4189-a66d-4ca34b42afe6.png)

After cutting appropriate time windows around the time of the reponse of the subjects, we get the epochs which we have to classify/decode:

![epoch_eeg](https://user-images.githubusercontent.com/28629116/170886309-80387b8e-2308-4e9e-ba01-b7b8333eaae3.png)

We also created three different levels of preprocessing to see how it affects the decoding:

![preprocessing](https://user-images.githubusercontent.com/28629116/170886754-1811ce79-128e-4569-985d-0fee4675357e.png)


## Training:

We trained our three architectures (EEGNet, Shallow ConvNet, and Deep ConvNet) on all seven paradigms, for three preprocessing levels, and on cross-validated random splits, within-subject and cross-subject. In addition, we also looked at different dataset sizes. All of this combined lead to quite a lot of permutations.

![training](https://user-images.githubusercontent.com/28629116/170886552-a94ab4b4-a5ca-43c0-8329-fa2797c80c95.png)

## Results:

After some simpler plots comparing the different deep learning models across paradigms:

![results1](https://user-images.githubusercontent.com/28629116/170887031-de43cdf0-b8bd-4d1e-9803-f5bcdfec2080.png)

We looked at cross- and within-subject data, where we had to plot our forty different subjects over seven paradigms to look for correlations:

![results_cross1](https://user-images.githubusercontent.com/28629116/170887851-a9fab8be-e2eb-47a3-afb8-889c67184dd8.png)

Interestingly, we didn't find any significant correlations of the subject performance between paradigms, going against the idea of "BCI Illiteracy":

![results_cross2](https://user-images.githubusercontent.com/28629116/170887918-fe8fcb1a-4735-4300-b7ad-c18c8f253966.png)

## Feature Attribution:

In order to visualize the decision-making of our different architectures, we looked at different feature attribution methods, and decided on DeepLift.
While there are significant problems with the interpretation of feature attribution methods, it nevertheless gave us insight into which electrodes and timeframes were important to our models.

![feature_attribution1](https://user-images.githubusercontent.com/28629116/170888182-544cd8d7-b549-4aee-8e78-bfdc7e7d5495.png)

The attribution values in most cases lined up well with the expected regions and timepoints of the respective ERP paradigms.

![feature_attribution2](https://user-images.githubusercontent.com/28629116/170888261-636d8022-ed1f-484e-8b69-b24601cea237.png)

## How to use:

Use DataLoader.py to load ERP CORE BIDS files (https://osf.io/9f5w7/), preprocess them, and save them as a dataframe.
Then use Training.py to initiate and train the Braindecode models. Examples on how to use them are in the DataLoader.ipynb and Training.ipynb notebooks.

The different experiments of the thesis are shown in ModelComparison.ipynb, SubjectAnalysis.ipynb, and FeatureAttribution.ipynb.
