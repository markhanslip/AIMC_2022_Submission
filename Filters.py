import parselmouth
import numpy as np
from numpy import inf
import statistics

class OnsetsFilter:

    def __init__(self, in_file, threshold=3, amp_thresh=50.0):
        self.in_file = in_file
        self.threshold = threshold
        self.audio_data = None
        self.pitches = None
        self.onsets = None
        self.amplitudes = None
        self.amp_thresh = amp_thresh

    def get_freqs(self):

        self.audio_data = parselmouth.Sound(self.in_file)
        # print('audio loaded')

        self.pitches = self.audio_data.to_pitch_ac(time_step=0.01, pitch_floor=50.0, pitch_ceiling=1200.0) # check this doesn't need a sr arg
        self.pitches = self.pitches.selected_array['frequency']
        self.pitches[self.pitches==0] = np.nan
        self.pitches = list(self.pitches)
        self.pitches = np.nan_to_num(self.pitches)
        # print('extracted freqs')

    def freqs_to_MIDI(self):

        self.pitches = 12*np.log2(self.pitches/440)+69
        self.pitches[self.pitches == -inf] = 0
        self.pitches = np.around(self.pitches, decimals=1)
        # print('converted freqs to MIDI')

    def get_onsets(self):

        # work out which note values represent an onset and multiply the two vectors
        temp_onsets = np.ediff1d(self.pitches) #or d = diff(midi)

        temp_onsets = (temp_onsets <= -0.8) & (temp_onsets >= -44) | (temp_onsets >= 0.8)
        temp_onsets = temp_onsets.astype(int)

        # replace consecutive onsets with 0:
        temp_onsets = list(temp_onsets)
        self.onsets=[]
        for i in range((len(temp_onsets)-1)):
            if temp_onsets[i] == 0:
                self.onsets.append(temp_onsets[i])
            if temp_onsets[i] == 1 and temp_onsets[i+1] == 0:
                self.onsets.append(temp_onsets[i])
            if temp_onsets[i] == 1 and temp_onsets[i+1] == 1:
                self.onsets.append(0)
        # print(len(self.onsets), len(self.pitches))
        self.onsets = np.insert(self.onsets, 0, 0)
        self.pitches = self.onsets * self.pitches[:-1]
        self.pitches[self.pitches > 80.0] == 0
        # print(self.pitches)
        # print(max(self.pitches))
        nz = np.flatnonzero(self.pitches)
        if np.count_nonzero(self.pitches) >= self.threshold: # 2 is number of meaningful freqs - set higher to make filter more aggressive
            return 1
        else:
            return 0

class AmpFilter:

    def __init__(self, in_file, threshold=35.0):
        self.in_file = in_file
        self.threshold = threshold
        self.audio_data = parselmouth.Sound(self.in_file)
        print('audio loaded')

    def get_mean_amp(self):

        self.amplitudes = self.audio_data.to_intensity(time_step=0.01)
        self.amplitudes = self.amplitudes.values
        self.amplitudes = np.ndarray.tolist(self.amplitudes)
        self.amplitudes = self.amplitudes[0]
        # print(self.amplitudes)
        if statistics.mean(self.amplitudes) > self.threshold:
            return 1
        else:
            return 0
