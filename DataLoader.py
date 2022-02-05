import mne, mne_bids, HelperFunctions, warnings
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from mne_bids import (BIDSPath, read_raw_bids)
from autoreject import AutoReject


def load_raw(data_path, task, preprocessing, subject_id):
    """
    Loads a single subject from the ERP Core data, applies filtering 
    and ICA, downsamples, and returns a mne.Raw object.
    """
    with HelperFunctions.suppress_stdout(), warnings.catch_warnings():
        # ignore warnings, as ERP Core is not quite in BIDS format
        warnings.simplefilter("ignore")
        bids_root = data_path+"/"+"ERP_CORE_BIDS_Raw_Files"
        bids_path = BIDSPath(subject=subject_id, task=task,
                             session=task, datatype='eeg', 
                             suffix='eeg', root=bids_root)
        raw = read_raw_bids(bids_path)
        raw.load_data()
        HelperFunctions.read_annotations_core(bids_path,raw)
        raw.set_channel_types({'HEOG_left': 'eog', 'HEOG_right': 'eog', 'VEOG_lower': 'eog'})
        raw.set_montage('standard_1020',match_case=False)
        raw = raw.resample(250)
        if task == "N170":
             raw = raw.set_eeg_reference(ref_channels='average')
        else: 
            raw = raw.set_eeg_reference(ref_channels=['P9', 'P10'])
        if preprocessing == "medium":
            raw.filter(0.5,40)
        if preprocessing == "heavy":
            raw.filter(0.5,40)
            ica, badComps = HelperFunctions.load_precomputed_ica(bids_root, subject_id,task)
            HelperFunctions.add_ica_info(raw,ica)
            ica.apply(raw)
    return raw

def epoch_raw(raw, task, preprocessing, reject_incorrect_responses=True):
    """
    Takes a mne.Raw object, loads the correct events, 
    and returns a mne.Epoch object.
    """
    with HelperFunctions.suppress_stdout():
        # get correct tmin, tmax, and event mapping per task
        custom_mapping, tmin, tmax = get_task_specifics(task)
        
        # shift annotations by lcd monitor delay
        if task != "MMN":
            raw.annotations.onset = raw.annotations.onset+.026
        
        # load events
        (events_from_annot, event_dict) = mne.events_from_annotations(raw, event_id=custom_mapping)
        
        if reject_incorrect_responses == True and task in ["N170", "N400", "N2pc", "P3"]:
        # only include events where response is in time and correct
            events_0, lag_0 = mne.event.define_target_events(events_from_annot, 0, 201, raw.info['sfreq'], 0, 0.8, 0, 999)
            events_1, lag_1 = mne.event.define_target_events(events_from_annot, 1, 201, raw.info['sfreq'], 0, 0.8, 1, 999)
            events_from_annot = np.concatenate((events_0, events_1), axis=0)
            # sort event array by timepoints to get rid of warning
            events_from_annot = events_from_annot[events_from_annot[:, 0].argsort()]
            # drop responses
            event_dict.pop('response:201')
            event_dict.pop('response:202')           

        # epoch with no constant detrend to remove dc offset, pick electrodes
        epoch = mne.Epochs(raw, events_from_annot, event_dict,
                           tmin=tmin,tmax=tmax, preload=True,
                           reject_by_annotation=True, baseline=None, 
                           picks=['FP1','F3','F7','FC3','C3','C5','P3','P7','P9','PO7',
                                  'PO3','O1','Oz','Pz','CPz','FP2','Fz','F4','F8','FC4',
                                  'FCz','Cz','C4','C6','P4','P8','P10','PO8','PO4','O2',], detrend=0)
        
        # apply autoreject for heavy preprocessing to remove artefacts
        if preprocessing == "heavy":
            ar = AutoReject()
            epoch = ar.fit_transform(epoch) 
    return epoch

def get_task_specifics(task, reject_incorrect_responses=True):
    """
    Returns mapping, tmin, tmax, specific to the task.
    """
    tmin = -0.2
    tmax = 0.8
    # mapping is always: 
    # ERP component not expected -> 0
    # ERP component expected -> 1
    # or left -> 0, right -> 1
    if task == "N170":
        # Cars: 0, Faces: 1
        custom_mapping = dict(("stimulus:"+str(i), 1) for i in range(0,41))
        custom_mapping.update(dict(("stimulus:"+str(i), 0) for i in range(41,81)))
    elif task == "N400":
        # unrelated word: 1, Related word: 0
        custom_mapping = {'stimulus:221': 1, 'stimulus:222': 1, 
                          'stimulus:211': 0, 'stimulus:212': 0} 
    elif task == "P3":
        # target=stimulus (rare): 1, target!=stimulus (frequent): 0
        custom_mapping = {'stimulus:11': 1, 'stimulus:12': 0, 'stimulus:13': 0, 'stimulus:14': 0, 'stimulus:15': 0, 
                          'stimulus:21': 0, 'stimulus:22': 1, 'stimulus:23': 0, 'stimulus:24': 0, 'stimulus:25': 0,
                          'stimulus:31': 0, 'stimulus:32': 0, 'stimulus:33': 1, 'stimulus:34': 0, 'stimulus:35': 0,
                          'stimulus:41': 0, 'stimulus:42': 0, 'stimulus:43': 0, 'stimulus:44': 1, 'stimulus:45': 0,
                          'stimulus:51': 0, 'stimulus:52': 0, 'stimulus:53': 0, 'stimulus:54': 0, 'stimulus:55': 1}
    elif task == "N2pc":
        # left: 0, right: 1
        custom_mapping = {'stimulus:111': 0, 'stimulus:112': 0, 'stimulus:211': 0, 'stimulus:212': 0, 
                          'stimulus:121': 1, 'stimulus:122': 1, 'stimulus:221': 1, 'stimulus:222': 1}
    elif task == "MMN":
        # deviant: 1, standard: 0
        custom_mapping = {'stimulus:70': 1, 'stimulus:80': 0}
    elif task == "ERN":
        # incorrect: 1, correct: 0
        custom_mapping = {'response:112': 1, 'response:122': 1, 'response:211': 1, 'response:221': 1,
                          'response:111': 0, 'response:121': 0, 'response:212': 0, 'response:222': 0}
        tmin = -0.6
        tmax = 0.4
    elif task == "LRP":
        # left response: 0, right response: 1
        custom_mapping = {'response:111': 0, 'response:112': 0, 'response:121': 0, 'response:122': 0, 
                          'response:211': 1, 'response:212': 1, 'response:221': 1, 'response:222': 1}
        tmin = -0.8
        tmax = 0.2
    # add button responses to tasks that have them
    if reject_incorrect_responses == True and task in ["N170", "N400", "N2pc", "P3"]:
            custom_mapping.update({'response:201': 201, 'response:202': 202})
    return custom_mapping, tmin, tmax

def create_df(data_path, task, preprocessing, n_subjects=40):
    """
    Creates combined dataframe with epoch, condition, subjectID and data.
    """
    df_list = []
    for i in range(n_subjects):
        subjectID = f"{i+1:03d}"
        raw = load_raw(data_path, task, preprocessing, subjectID)
        epoch = epoch_raw(raw, task, preprocessing)
        df = epoch.to_data_frame()
        df["subjectID"] = i
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    
    # change condition naming to binary labels
    custom_mapping = get_task_specifics(task)[0]
    for condition in custom_mapping:
        df["condition"] = df["condition"].replace(condition,custom_mapping[condition])
    
    # reshape data
    window_size = 251
    data = df.iloc[:,3:33]
    data = data.to_numpy().reshape(int(data.shape[0]/window_size), window_size, -1)
    data = np.transpose(data,axes=[0,2,1])
    # create labels
    df = df[["epoch","condition","subjectID"]].drop_duplicates()
    df = df.reset_index()
    
    df["data"]=pd.Series(list(data))
    df = df.drop(columns=["index"])
    return df

def save_df(df):
    df.to_pickle(data_path+"/Dataframes/"+task+"_"+preprocessing+".pkl")

def load_df(data_path, task, preprocessing):
    df = pd.read_pickle(data_path+"/Dataframes/"+task+"_"+preprocessing+".pkl")
    return df

def create_data_labels(df, list_of_subjects=None):
    """
    Takes dataframe and returns numpy versions of the data and labels. 
    """
    # get data from dataframe and reshape back
    if list_of_subjects != None:
        df = df[df["subjectID"].isin(list_of_subjects)]
    data = np.dstack(df["data"].to_numpy())
    data = np.moveaxis(data, -1, 0)
    # get labels from dataframe
    labels = df["condition"].to_numpy()
    
    return data, labels