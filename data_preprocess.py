import os

import mne
import numpy as np
import hdf5storage
import matplotlib
from mne.io import Raw
# from mne.time_frequency import psd_welch

ch_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
            '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54',
            '55', '56', '57', '58', '59'
                                    '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
            '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',
            '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
            ]  # Channel names

FREQ_BANDS = [
    {'name': 'Delta-Theta', 'fmin': 1, 'fmax': 8},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 12},
    {'name': 'Beta', 'fmin': 12, 'fmax': 30},
    {'name': 'Gamma', 'fmin': 30, 'fmax': 100}
]

# For each subject, process one subject at a time
path = 'dataset/iEEG_CorrelationsWithMood/' # Replace with your actual path
# path_list = os.listdir(path)

path_list = [
    'EC108_data.mat',
    'EC113_data.mat',
    'EC122_data.mat',
    'EC125_data.mat',
    'EC129_data.mat',
    'EC133_data.mat',
    'EC137_data.mat',
    'EC139_data.mat',
    'EC142_data.mat',
    'EC150_data.mat',
    'EC152_data.mat',
    'EC153_data.mat',
    'EC162_data.mat',
    'EC175_data.mat',
    'EC81_data.mat',
    'EC82_data.mat',
    'EC84_data.mat',
    'EC87_data.mat',
    'EC91_data.mat',
    'EC92_data.mat',
    'EC96_data.mat',
    'EC99_data.mat'
]

# path_list = ['EC113_data.mat']
for j in path_list:
    print('subject: {}'.format(j))
    file = hdf5storage.loadmat(path + j)

    data_ECoG = file['ECoG']
    label_IMS = file['IMS']

    # Split into multiple 5-minute segments according to IMS
    data = []
    data_temp = []
    for i in data_ECoG:
        data_i = i[0][0]
        # Shape of each ECoG data
        data_shape = data_i.shape[0]
        ch_num = data_i.shape[1]
        # print(data_shape)
        # Corresponding sampling rate
        fs = i[0][2][0][0]
        # print(fs)

        # Downsample, record data_i (i[0][0])
        data_i = data_i.transpose()
        info = mne.create_info(ch_names[:ch_num], fs)

        raw = mne.io.RawArray(data_i, info)
        sfreq = 512
        raw_downsampled = raw.copy().resample(sfreq=sfreq)
        data_i = raw_downsampled.get_data()

        # print(data_i.shape)
        # data_i is the actual data
        data_temp.append(data_i)
        if len(data_temp) == 1:
            interval = data_i.shape[1] / sfreq
            # print(interval)
            if interval > 292:
                data.append(data_i)
                data_temp.clear()
            else:
                continue
        else:
            # Merge numpy arrays
            interval = 0
            for x in data_temp:
                interval += x.shape[1] / sfreq

            if interval > 275:
                if len(data_temp) == 2:
                    temp = np.concatenate((data_temp[0], data_temp[1]), axis=1)

                else:
                    temp = np.concatenate((data_temp[0], data_temp[1]), axis=1)
                    for d in data_temp[2:]:
                        temp = np.concatenate((temp, d), axis=1)

                data.append(temp)
                data_temp.clear()

            else:
                continue

    # temp = np.concatenate((data_temp[0], data_temp[1]), axis=1)
    # data_1_1.append(temp)
    # data_temp.clear()
    # print()
    # for i in range(len(data_1_1)):
    #     print(data_1_1[i].shape)
    # print(len(data_1_1))
    # print(len(label_IMS))

    # Divide into trials every 5 seconds, using sfreq = 512Hz

    trial_time = 5
    F = trial_time * sfreq
    num_trial = len(data)
    for i in range(num_trial):
        # print(data_1_1[0].shape)
        trial_num = data[i].shape[1] // F
        # print(trial_num)
        # Remove extra time points
        new_arr = data[i][:, :trial_num * F].reshape(ch_num, F, trial_num)
        # print(new_arr.shape)
        new_arr = new_arr.transpose(2, 0, 1)
        # print(new_arr.shape)
        data[i] = new_arr

    label = []
    # Get the final processed data for each subject
    for i in range(len(label_IMS)):
        label_i = label_IMS[i]
        # print(label_i[0][0][0])
        # print(label_i[0][0][0].shape)

        new_label = label_i[0][0][0][0]
        new_label = np.expand_dims(new_label, axis=-1)
        # print(new_label.shape)
        label.append(new_label)

    # print(label)

    # data_1_1 = np.vstack(data_1_1)
    # label = np.vstack(label)
    # data_1_1 = np.array(data_1_1)
    label = np.array(label)

    print(len(data))
    print(len(label))

    save_path = 'dataset/processed_data/' # Replace with your actual save path
    save_name = j.split('_')[0]
    np.savez(save_path + save_name, x=data, y=label)