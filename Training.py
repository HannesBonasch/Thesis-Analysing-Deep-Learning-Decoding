import torch, skorch, sklearn, os, json, DataLoader
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

def init_model(model_name, lr, n_epochs=25, batch_size=64, n_chan=30, 
               n_classes=2, weight_decay=0, seed=42, 
               input_window_samples=251, valid_ds=None, class_weights=None,
               gpu=True):
    """
    Initializes the model and classifier.
    """
    if gpu and torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
        # set seed for reproducability
        set_random_seeds(seed=seed, cuda=True)
    else:
        device = 'cpu'
        set_random_seeds(seed=seed, cuda=False)
    
    # load model
    if model_name == "eegnet":
        model = EEGNetv4(
            n_chan,
            n_classes,
            input_window_samples=input_window_samples,
            final_conv_length="auto",
        )
    elif model_name == "shallow":
        model = ShallowFBCSPNet(
            n_chan,
            n_classes,
            input_window_samples=input_window_samples,
            n_filters_time=40, 
            filter_time_length=25, 
            n_filters_spat=40, 
            pool_time_length=75, 
            pool_time_stride=15, 
            final_conv_length="auto"
            
        )
    elif model_name == "deep":
        model = Deep4Net(
            n_chan,
            n_classes,
            input_window_samples=input_window_samples,
            n_filters_time=25, 
            n_filters_spat=25, 
            filter_time_length=10,
            # changed stride to fit shorter input
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
    elif model_name == "tcn":
        model = TCN(
            n_chan,
            n_classes,
            n_filters=50,
            n_blocks=7,
            kernel_size=2,
            drop_prob=0.3,
            add_log_softmax=True
        )
    
    # send model to gpu
    if device == 'cuda':
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
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            #"accuracy",
            #"balanced_accuracy",
            #"roc_auc",
            ("train_balanced_accuracy", skorch.callbacks.EpochScoring(scoring='balanced_accuracy', on_train=True, name="train_balanced_accuracy", lower_is_better=False)),
            ("valid_balanced_accuracy", skorch.callbacks.EpochScoring(scoring='balanced_accuracy', on_train=False, name="valid_balanced_accuracy", lower_is_better=False)),
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )
    clf.initialize()
    # number of trainable parameters
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    return clf, model

def run_exp(data, labels, task, preprocessing, model_folder, model_name, 
            lr, n_epochs, n_splits, batch_size=64, additional_save_param=""):
    """
    Trains classifier using Stratified Cross Validation and saves parameters and history.
    """
    # path to save to
    model_path = os.getcwd()+"\\"+model_folder+"\\"+model_name+"\\"+task+"\\"+preprocessing+"\\"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    # calculate class weights
    class_weights=class_weight.compute_class_weight('balanced',np.unique(labels),labels)
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    class_weights = class_weights.to('cuda')
    
    # push data and labels to gpu
    dataset = TensorDataset(torch.from_numpy(data).to('cuda'),
                            torch.from_numpy(labels).to('cuda'))
    
    # create stratified splits
    cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits, test_size=0.2, random_state=42)
    cv_split = cv.split(data,labels)

    # train and validate on each split, then save parameters and history
    i = 0
    for train_idx, test_idx in cv_split:
        i += 1
        #valid_ds = TensorDataset(torch.from_numpy(data[test_idx]), torch.from_numpy(labels[test_idx]))
        clf, model = init_model(model_name, lr, n_epochs=25, batch_size=64, n_chan=30, 
               n_classes=2, weight_decay=0, seed=42, input_window_samples=251, 
               valid_ds=torch.utils.data.Subset(dataset, test_idx), 
               class_weights=class_weights, gpu=True)
        #clf, model = init_model(model_name, lr, n_epochs, batch_size, 
        #                        valid_ds=torch.utils.data.Subset(dataset, test_idx), 
        #                        class_weights=class_weights)  
        clf.fit(torch.utils.data.Subset(dataset, train_idx), y=None, epochs=n_epochs)
        clf.save_params(f_params=model_path+"split_"+str(i)+additional_save_param+"_model.pkl",
                       f_optimizer=model_path+"split_"+str(i)+additional_save_param+"_optimizer.pkl",
                       f_history=model_path+"split_"+str(i)+additional_save_param+"_history.json")
        
def load_exp(model_folder, model_name, task, preprocessing, n_splits, model_path=None, additional_save_param=""):
    """
    Loads the history json and puts it in a dataframe.
    """
    if model_path == None:
        model_path = os.getcwd()+"\\"+model_folder+"\\"+model_name+"\\"+task+"\\"+preprocessing+"\\"
    df_list = []
    for i in range(1,n_splits+1):
        df_list.append(pd.read_json(model_path+"split_"+str(i)+additional_save_param+"_history.json"))
    df = pd.concat(df_list,axis=1)
    
    return df

def run_exp_per_subject(df, task, preprocessing, model_folder, model_name, 
            lr, n_epochs, batch_size=64, n_subjects=40):
    """
    Trains classifier on all but one subject and saves parameters and history.
    """
    # path to save to
    model_path = os.getcwd()+"\\"+model_folder+"\\"+model_name+"\\"+task+"\\"+preprocessing+"\\"
    Path(model_path).mkdir(parents=True, exist_ok=True)
        
    # train and validate on each subject, then save parameters and history
    for i in range(n_subjects):
        list_train = list(range(n_subjects))
        list_train.remove(i)
        data, labels = DataLoader.create_data_labels(df, list_train)
        # calculate class weights
        class_weights=class_weight.compute_class_weight('balanced',np.unique(labels),labels)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        class_weights = class_weights.to('cuda')
        # push data and labels to gpu
        dataset = TensorDataset(torch.from_numpy(data).to('cuda'),
                                torch.from_numpy(labels).to('cuda'))
        
        valid_data, valid_labels = DataLoader.create_data_labels(df, [i])
        valid_dataset = TensorDataset(torch.from_numpy(valid_data).to('cuda'),
                                      torch.from_numpy(valid_labels).to('cuda'))
        
        clf, model = init_model(model_name, lr, n_epochs=25, batch_size=64, n_chan=30, 
                               n_classes=2, weight_decay=0, seed=42, input_window_samples=251, 
                               valid_ds=valid_dataset, 
                               class_weights=class_weights, gpu=True)
        clf.fit(dataset, y=None, epochs=n_epochs)
        clf.save_params(f_params=model_path+"split_"+str(i)+"_model.pkl",
                       f_optimizer=model_path+"split_"+str(i)+"_optimizer.pkl",
                       f_history=model_path+"split_"+str(i)+"_history.json")
 
def run_exp_single_subject(df, task, preprocessing, model_folder, model_name, 
            lr, n_epochs, n_splits, batch_size=64, n_subjects=40):
    """
    Trains classifier on single subject and saves parameters and history.
    """
    # path to save parameters to
    model_path = os.getcwd()+"\\"+model_folder+"\\"+model_name+"\\"+task+"\\"+preprocessing+"\\"
    Path(model_path).mkdir(parents=True, exist_ok=True)
        
    # train and validate on each subject, then save parameters and history
    for subjectID in range(n_subjects):
        data, labels = DataLoader.create_data_labels(df, [subjectID])
        # calculate class weights
        class_weights=class_weight.compute_class_weight('balanced',np.unique(labels),labels)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        class_weights = class_weights.to('cuda')
        # push data and labels to gpu
        dataset = TensorDataset(torch.from_numpy(data).to('cuda'),
                                torch.from_numpy(labels).to('cuda'))
        
        
        
        # create stratified splits
        cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits, test_size=0.2, random_state=42)
        cv_split = cv.split(data,labels)

        # train and validate on each split, then save parameters and history
        i = 0
        for train_idx, test_idx in cv_split:
            i += 1
            #valid_ds = TensorDataset(torch.from_numpy(data[test_idx]), torch.from_numpy(labels[test_idx]))
            clf, model = init_model(model_name, lr, n_epochs=25, batch_size=64, n_chan=30, 
                               n_classes=2, weight_decay=0, seed=42, input_window_samples=251, 
                               valid_ds=torch.utils.data.Subset(dataset, test_idx), 
                               class_weights=class_weights, gpu=True)
            clf.fit(torch.utils.data.Subset(dataset, train_idx), y=None, epochs=n_epochs)
            clf.save_params(f_params=model_path+"subject_"+str(subjectID)+"_split_"+str(i)+"_model.pkl",
                           f_optimizer=model_path+"subject_"+str(subjectID)+"_split_"+str(i)+"_optimizer.pkl",
                           f_history=model_path+"subject_"+str(subjectID)+"_split_"+str(i)+"_history.json")