import HelperFunctions
import mne
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import TensorDataset
from mne.event import define_target_events


def load_data(parameters):
    """
    Loads the ERP Core Data and returns a list of mne.Raw objects.
    """
    # get correct preprocessing file names
    if parameters["preprocessing"]=='light':
        preprocessing = "_shifted_ds_reref_ucbip.set"
    elif parameters["preprocessing"]=='medium':
        preprocessing = "_shifted_ds_reref_ucbip_hpfilt.set"
    elif parameters["preprocessing"]=='heavy':
        preprocessing = "_shifted_ds_reref_ucbip_hpfilt_ica_weighted.set"
    else:
        print("Wrong Preprocessing")    
    if parameters["task"]== "MMN":
        # cut shifted for MMN
        preprocessing = preprocessing[8:]
        
    # first create list of data paths, then read data with mne
    erp_core_paths = []    
    for i in range(1,parameters["n_subjects"]+1):
        erp_core_paths.append(parameters["data_path"] + "/"+parameters["task"]+"/"+str(i)+"/"+str(i)+"_"
                              +parameters["task"]+preprocessing)
    list_of_raws = [mne.io.read_raw_eeglab(path, preload=True) for path in erp_core_paths]
    
    return list_of_raws


def epoch_raws(list_of_raws, parameters):
    # TODO: This could be a bit cleaner.
    """
    Takes a list of mne.Raw objects, epochs them using the correct tmin, tmax, 
    and events for each task, and then returns a dataframe containing all epochs.
    """   
    # get correct tmin, tmax, and event mapping per task
    tmin = -0.2
    tmax = 0.8
    if parameters["task"] == "N170":
        # Cars: 0, Faces: 1
        custom_mapping = dict((str(i), 1) for i in range(0,41))
        custom_mapping.update(dict((str(i), 0) for i in range(41,81)))
    elif parameters["task"] == "N400":
        # unrelated word: 0, Related word: 1
        custom_mapping = {'221': 0, '222': 0, '211': 1, '212': 1} 
    elif parameters["task"] == "N2pc":
        # left: 0, right: 1
        custom_mapping = {'111': 0, '112': 0, '211': 0, '212': 0, 
                          '121': 1, '122': 1, '221': 1, '222': 1}
    elif parameters["task"] == "P3":
        # target=stimulus (rare): 0, target!=stimulus (frequent): 1
        custom_mapping = {'11': 0, '12': 1, '13': 1, '14': 1, '15': 1, 
                          '21': 1, '22': 0, '23': 1, '24': 1, '25': 1,
                          '31': 1, '32': 1, '33': 0, '34': 1, '35': 1,
                          '41': 1, '42': 1, '43': 1, '44': 0, '45': 1,
                          '51': 1, '52': 1, '53': 1, '54': 1, '55': 0}
    elif parameters["task"] == "MMN":
        # deviant: 0, standard: 1
        custom_mapping = {'70': 0, '80': 1}
    elif parameters["task"] == "ERN":
        # incorrect: 0, correct: 1
        custom_mapping = {'112': 0, '122': 0, '211': 0, '221': 0,
                          '111': 1, '121': 1, '212': 1, '222': 1}
        tmin = -0.6
        tmax = 0.4
    elif parameters["task"] == "LRP":
        # left response: 0, right response: 1
        custom_mapping = {'111': 0, '112': 0, '121': 0, '122': 0, 
                          '211': 1, '212': 1, '221': 1, '222': 1}
        tmin = -0.8
        tmax = 0.2

    # create list of epoch dataframes including all subjects
    list_of_epochs = []
    for i in range(0,parameters["n_subjects"]):
        # add responses
        if parameters["reject_incorrect_responses"] == True and parameters["task"] in ["N170", "N400", "N2pc", "P3"]:
            custom_mapping.update({'201': 201, '202': 202})


        (events_from_annot, event_dict) = mne.events_from_annotations(list_of_raws[i], 
                                                                      event_id=custom_mapping)
        # TODO: Sanity checks on incorrect responses
        # TODO: Sort event list after combining
        # only include events where response is in time and correct
        if parameters["reject_incorrect_responses"] == True and parameters["task"] in ["N170", "N400", "N2pc", "P3"]:
            events_0, lag_0 = define_target_events(events_from_annot, 0, 201, list_of_raws[0].info['sfreq'], 0, 0.8, 0, 999)
            events_1, lag_1 = define_target_events(events_from_annot, 1, 201, list_of_raws[0].info['sfreq'], 0, 0.8, 1, 999)
            events_from_annot = np.concatenate((events_0, events_1), axis=0)
            event_dict.pop("201")
            event_dict.pop("202")
        
        epoch = mne.Epochs(list_of_raws[i], events_from_annot, event_dict,tmin=tmin,tmax=tmax, 
                           reject_by_annotation=False, baseline=(None,0), picks=range(0,28))
        epoch = epoch.to_data_frame()
        epoch["subjectID"]=i+1
        list_of_epochs.append(epoch)
        
    df_epochs = pd.concat(list_of_epochs, axis=0)
    
    # change condition naming to binary labels
    for condition in custom_mapping:
        df_epochs["condition"] = df_epochs["condition"].replace(condition,custom_mapping[condition])
        
    return df_epochs


def load_dataframe(parameters):
    # TODO: This might be better with df_epochs as an input instead of running load_data and epoch_raws
    """
    Loads data, epochs it, then reshapes the dataframe into data and labels.
    """
     # load data and epoch
    list_of_raws = load_data(parameters)
    df_epochs = epoch_raws(list_of_raws, parameters)
    
    # TODO: Get rid of fixed number for unique timepoints
    # reshape data
    data = df_epochs.iloc[:,3:31]
    #data = (df_epochs.iloc[:,3:31]-df_epochs.iloc[:,3:31].mean())/df_epochs.iloc[:,3:31].std()
    data = data.to_numpy().reshape(int(data.shape[0]/257), 257, -1)
    data = np.transpose(data,axes=[0,2,1])
    
    # create labels
    df = df_epochs[["epoch","condition","subjectID"]].drop_duplicates()
    df = df.reset_index()
    
    df["data"]=pd.Series(list(data))
    df = df.drop(columns=["index"])
    
    return df


def create_data_labels(df):
    """
    Takes dataframe and returns numpy versions of the data and labels. 
    """
    # get data from dataframe and reshape back
    data = np.dstack(df["data"].to_numpy())
    data = np.moveaxis(data, -1, 0)
    # get labels from dataframe
    labels = df["condition"].to_numpy()
    
    return data, labels


def create_dataset(df):
    """
    Takes dataframe and returns pytorch dataset. 
    """
    data, labels = create_data_labels(df)
    # create dataset
    dataset = TensorDataset(torch.from_numpy(data).float(),torch.from_numpy(labels).float())
    
    return dataset