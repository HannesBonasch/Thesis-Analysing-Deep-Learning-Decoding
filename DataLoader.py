import mne, mne_bids, HelperFunctions
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from mne_bids import (BIDSPath, read_raw_bids)
from autoreject import AutoReject

def load_raw(parameters, subject_id):
    """
    Loads a single subject from the ERP Core data, applies filtering and ICA, and returns a mne.Raw object.
    """
    with HelperFunctions.suppress_stdout():
        bids_root = parameters["data_path"]+"/"+parameters["task"]
        bids_path = BIDSPath(subject=subject_id, task=parameters["task"],
                             session=parameters["task"], datatype='eeg', 
                             suffix='eeg', root=bids_root)
        raw = read_raw_bids(bids_path)
        raw.load_data()
        HelperFunctions.read_annotations_core(bids_path,raw)
        raw.set_channel_types({'HEOG_left': 'eog', 'HEOG_right': 'eog', 'VEOG_lower': 'eog'})
        raw.set_montage('standard_1020',match_case=False)
        if parameters["preprocessing"] == "medium":
            raw.filter(0.5,40)
        if parameters["preprocessing"] == "heavy":
            raw.filter(0.5,40)
            ica, badComps = HelperFunctions.load_precomputed_ica(bids_root, subject_id,parameters["task"])
            HelperFunctions.add_ica_info(raw,ica)
            ica.apply(raw)
    return raw

def epoch_raw(parameters, raw):
    """
    Takes a mne.Raw object, loads the correct events, and returns a mne.Epoch object.
    """
    with HelperFunctions.suppress_stdout():
        # get correct tmin, tmax, and event mapping per task
        custom_mapping, tmin, tmax = get_task_specifics(parameters)
        
        # shift annotations by lcd monitor delay
        if parameters["task"] != "MNE":
            raw.annotations.onset = raw.annotations.onset+.026
        
        # load events
        (events_from_annot, event_dict) = mne.events_from_annotations(raw, event_id=custom_mapping)
        
        if parameters["reject_incorrect_responses"] == True and parameters["task"] in ["N170", "N400", "N2pc", "P3"]:
        # only include events where response is in time and correct
            events_0, lag_0 = mne.event.define_target_events(events_from_annot, 0, 201, raw.info['sfreq'], 0, 0.8, 0, 999)
            events_1, lag_1 = mne.event.define_target_events(events_from_annot, 1, 201, raw.info['sfreq'], 0, 0.8, 1, 999)
            events_from_annot = np.concatenate((events_0, events_1), axis=0)
            # sort event array by timepoints to get rid of warning
            events_from_annot = events_from_annot[events_from_annot[:, 0].argsort()]
            # drop responses
            event_dict.pop('response:201')
            event_dict.pop('response:202')           

        # epoch with no constant detrend to remove dc offset
        epoch = mne.Epochs(raw, events_from_annot, event_dict,
                           tmin=tmin,tmax=tmax, preload=True,
                           reject_by_annotation=True, baseline=None, 
                           picks=range(0,30), detrend=0)
        
        # apply autoreject for heavy preprocessing to remove artefacts
        if parameters["preprocessing"] == "heavy":
            ar = AutoReject()
            epoch = ar.fit_transform(epoch) 
    return epoch

def get_task_specifics(parameters):
    """
    Returns mapping, tmin, tmax, specific to the task.
    """
    tmin = -0.2
    tmax = 0.8
    if parameters["task"] == "N170":
        # Cars: 0, Faces: 1
        custom_mapping = dict(("stimulus:"+str(i), 1) for i in range(0,41))
        custom_mapping.update(dict(("stimulus:"+str(i), 0) for i in range(41,81)))
    elif parameters["task"] == "N400":
        # unrelated word: 0, Related word: 1
        custom_mapping = {'stimulus:221': 0, 'stimulus:222': 0, 
                          'stimulus:211': 1, 'stimulus:212': 1} 
    elif parameters["task"] == "P3":
        # target=stimulus (rare): 0, target!=stimulus (frequent): 1
        custom_mapping = {'stimulus:11': 0, 'stimulus:12': 1, 'stimulus:13': 1, 'stimulus:14': 1, 'stimulus:15': 1, 
                          'stimulus:21': 1, 'stimulus:22': 0, 'stimulus:23': 1, 'stimulus:24': 1, 'stimulus:25': 1,
                          'stimulus:31': 1, 'stimulus:32': 1, 'stimulus:33': 0, 'stimulus:34': 1, 'stimulus:35': 1,
                          'stimulus:41': 1, 'stimulus:42': 1, 'stimulus:43': 1, 'stimulus:44': 0, 'stimulus:45': 1,
                          'stimulus:51': 1, 'stimulus:52': 1, 'stimulus:53': 1, 'stimulus:54': 1, 'stimulus:55': 0}
    # add button responses to tasks that have them
    if parameters["reject_incorrect_responses"] == True and parameters["task"] in ["N170", "N400", "N2pc", "P3"]:
            custom_mapping.update({'response:201': 201, 'response:202': 202})
    return custom_mapping, tmin, tmax

def create_df(parameters):
    df_list = []
    for i in range(parameters["n_subjects"]):
        subjectID = f"{i+1:03d}"
        raw = load_raw(parameters, subjectID)
        epoch = epoch_raw(parameters, raw)
        df = epoch.to_data_frame()
        df["subjectID"] = i
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    
    # change condition naming to binary labels
    custom_mapping = get_task_specifics(parameters)[0]
    for condition in custom_mapping:
        df["condition"] = df["condition"].replace(condition,custom_mapping[condition])
    
    # reshape data
    data = df.iloc[:,3:33]
    data = data.to_numpy().reshape(int(data.shape[0]/parameters["input_window_samples"]), parameters["input_window_samples"], -1)
    data = np.transpose(data,axes=[0,2,1])
    # create labels
    df = df[["epoch","condition","subjectID"]].drop_duplicates()
    df = df.reset_index()
    
    df["data"]=pd.Series(list(data))
    df = df.drop(columns=["index"])
    return df

def load_df(parameters):
    df = pd.read_pickle(parameters["data_path"]+"/Dataframes/"+parameters["task"]+"_"+parameters["preprocessing"]+".pkl")
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