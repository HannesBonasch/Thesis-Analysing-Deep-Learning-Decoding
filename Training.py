import DataLoader
import torch
import sklearn
import os
import json
import numpy as np
import pandas as pd
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, EEGNetv1, Deep4Net, TCN, EEGNetv4
from skorch.callbacks import LRScheduler
from skorch.dataset import CVSplit
from braindecode import EEGClassifier
from torch.utils.data import TensorDataset
from skorch.helper import predefined_split
from pathlib import Path
from sklearn.utils import class_weight


def init_model(parameters, valid_ds=None, class_weights=None):
    # TODO: Add more models.
    """
    Initializes the model and classifier depending on the parameters.
    """
    # choosing gpu if possible
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    # set seed for reproducability
    set_random_seeds(seed=parameters["seed"], cuda=cuda)
    
    
    # load model
    if parameters["model"] == "eegnet":
        model = EEGNetv4(
            parameters["n_chans"],
            parameters["n_classes"],
            input_window_samples=parameters["input_window_samples"],
            final_conv_length="auto",
        )
    elif parameters["model"] == "shallow":
        model = ShallowFBCSPNet(
            parameters["n_chans"],
            parameters["n_classes"],
            input_window_samples=parameters["input_window_samples"],
            n_filters_time=40, 
            filter_time_length=25, 
            n_filters_spat=40, 
            pool_time_length=75, 
            pool_time_stride=15, 
            final_conv_length="auto"
            
        )
    elif parameters["model"] == "deep":
        model = Deep4Net(
            parameters["n_chans"],
            parameters["n_classes"],
            input_window_samples=parameters["input_window_samples"],
            n_filters_time=25, 
            n_filters_spat=25, 
            filter_time_length=10,
            # TODO: changed stride, need to better figure out why it's needed
            pool_time_length=2, 
            pool_time_stride=2, 
            n_filters_2=50, 
            filter_length_2=10, 
            n_filters_3=100, 
            filter_length_3=10, 
            n_filters_4=200, 
            filter_length_4=10,
            final_conv_length="auto",
        )
    elif parameters["model"] == "tcn":
        model = TCN(
            parameters["n_chans"],
            parameters["n_classes"],
            n_filters=15,
            n_blocks=7,
            kernel_size=2,
            drop_prob=0.3,
            add_log_softmax=True
        )
    # send model to gpu
    if cuda:
        model.cuda()
        
    if valid_ds==None:
        train_split=None
    else:
        train_split=predefined_split(valid_ds)
    
    # load classifier
    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        criterion__weight=class_weights,
        optimizer=torch.optim.AdamW,
        train_split=train_split,
        optimizer__lr=parameters["lr"],
        optimizer__weight_decay=parameters["weight_decay"],
        batch_size=parameters["batch_size"],
        callbacks=[
            "accuracy",
            "balanced_accuracy",
            #"roc_auc",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=parameters["n_epochs"] - 1)),
        ],
        device=device,
    )
    clf.initialize()
    # number of trainable parameters
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    return clf

def run_exp(data, labels, parameters):
    """
    Trains classifier using Stratified Cross Validation and saves parameters and history.
    """
    # path to save parameters to
    model_path = os.getcwd()+"\\"+parameters["model_folder"]+"\\"+parameters[
        "model"]+"\\"+parameters["task"]+"\\"+parameters["preprocessing"]+"\\"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    json.dump(parameters, open(model_path+"parameters.json", 'w' ))
    
    # calculate class weights
    class_weights=class_weight.compute_class_weight('balanced',np.unique(labels),labels)
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    
    # create stratified splits
    cv = sklearn.model_selection.StratifiedShuffleSplit(parameters["n_splits"], test_size=0.2, random_state=42)
    cv_split = cv.split(data,labels)

    # train and validate on each split, then save parameters and history
    i = 0
    for train_idx, test_idx in cv_split:
        i += 1
        valid_ds = TensorDataset(torch.from_numpy(data[test_idx]), torch.from_numpy(labels[test_idx]))
        clf = init_model(parameters, valid_ds, class_weights)
        clf.fit(data[train_idx], y=labels[train_idx], epochs=parameters["n_epochs"])
        clf.save_params(f_params=model_path+"split_"+str(i)+"_model.pkl",
                       f_optimizer=model_path+"split_"+str(i)+"_optimizer.pkl",
                       f_history=model_path+"split_"+str(i)+"_history.json")
        
def load_exp(parameters):
    """
    Loads the history json and puts it in a dataframe.
    """
    model_path = os.getcwd()+"\\"+parameters["model_folder"]+"\\"+parameters[
        "model"]+"\\"+parameters["task"]+"\\"+parameters["preprocessing"]+"\\"
    df_list = []
    for i in range(1,parameters["n_splits"]+1):
        df_list.append(pd.read_json(model_path+"split_"+str(i)+"_history.json"))
    df = pd.concat(df_list,axis=1)
    
    return df