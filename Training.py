import DataLoader
import torch
import sklearn
import os
import pandas as pd
from torch.utils.data import TensorDataset
from skorch.helper import predefined_split
from pathlib import Path
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, EEGNetv1
from skorch.callbacks import LRScheduler
from skorch.dataset import CVSplit


def init_model(parameters, valid_ds=None):
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
    model = EEGNetv1(
        parameters["n_chans"],
        parameters["n_classes"],
        input_window_samples=parameters["input_window_samples"],
        final_conv_length='auto',
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
        optimizer=torch.optim.AdamW,
        train_split=train_split,
        optimizer__lr=parameters["lr"],
        optimizer__weight_decay=parameters["weight_decay"],
        batch_size=parameters["batch_size"],
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=parameters["n_epochs"] - 1)),
        ],
        device=device,
    )
    clf.initialize()
    
    return clf

def run_exp(data, labels, parameters):
    # TODO: Add metrics and class weights for unbalanced datasets.
    """
    Trains classifier using Stratified Cross Validation and saves parameters and history.
    """
    # path to save parameters to
    model_path = os.getcwd()+"\\"+parameters["model_folder"]+"\\"+parameters[
        "model"]+"\\"+parameters["task"]+"\\"+parameters["preprocessing"]+"\\"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    # create stratified splits
    cv = sklearn.model_selection.StratifiedShuffleSplit(parameters["n_splits"], test_size=0.2, random_state=42)
    cv_split = cv.split(data,labels)

    # train and validate on each split, then save parameters and history
    i = 0
    for train_idx, test_idx in cv_split:
        i += 1
        valid_ds = TensorDataset(torch.from_numpy(data[test_idx]), torch.from_numpy(labels[test_idx]))
        clf = init_model(parameters, valid_ds)
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