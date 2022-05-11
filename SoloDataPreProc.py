#!/usr/bin/env python

import os
import numpy as np
import librosa
import scipy.io.wavfile as wav
import soundfile as sf
import skimage.io as io
import sox
from shutil import rmtree, move
from pydub import AudioSegment, effects
from numpy import inf
import parselmouth
import torch
import torchaudio
from nnAudio import features
from rich import print

class SoloDataProcessing:

    def __init__(self, name: str, in_file: str): # change to accept a category name only

        self.classname = name
        self.in_file = in_file
        self.chunk_dir = './{}_chunks'.format(self.classname)
        self.aug_dir = './{}_aug'.format(self.classname)
        self.sax_train_dir = './sax_train_data'
        self.cqt_dir = os.path.join(self.sax_train_dir, './{}'.format(self.classname))
        self.phrase_dir = './{}_phrases'.format(self.classname)
        self.d_phrase_dir = './{}_d_phrases'.format(self.classname)
        self.sr=22050
        #if self.classname != 'silence':
        self.silent_file = './{}_silence.wav'.format(self.classname)
        self.pitches = None
        self.pitch_thresh = None
        self.cqt = None

    def make_dirs(self):

        if not os.path.exists(self.sax_train_dir):
            os.mkdir(self.sax_train_dir)
        if not os.path.exists(self.chunk_dir):
            os.mkdir(self.chunk_dir)
        if not os.path.exists(self.aug_dir):
            os.mkdir(self.aug_dir)
        if not os.path.exists(self.cqt_dir):
            os.mkdir(self.cqt_dir)
        if not os.path.exists(self.phrase_dir):
            os.mkdir(self.phrase_dir)

        print('created directories')

    def resample_audio(self):
        sr, data = wav.read(self.in_file)
        data = data.astype(np.float64)
        if sr != 22050:
            data = librosa.resample(data, sr, self.sr, 'polyphase')
            wav.write(self.in_file, self.sr, data)
            print('resampled source audio')
        else:
            print('data is already at desired sr of 22050')
            pass

    def normalize(self):
        # safeguard
        sr, data = wav.read(self.in_file)
        data = data.astype(np.int16)
        wav.write(self.in_file, self.sr, data)

        rawsound = AudioSegment.from_file(self.in_file, "wav")
        normalizedsound = effects.normalize(rawsound)
        normalizedsound.export(self.in_file, format="wav")
        print('normalized audio')

    def chunk_by_phrase(self):

        audio_data = parselmouth.Sound(self.in_file)
        print('audio loaded')
        self.pitches = audio_data.to_pitch_ac(time_step=0.1, pitch_floor=50.0, pitch_ceiling=1400.0) # check this doesn't need a sr arg
        self.pitches = self.pitches.selected_array['frequency']
        self.pitches[self.pitches==0] = np.nan
        # self.pitches = list(self.pitches)
        self.pitches = np.nan_to_num(self.pitches)
        self.pitches=list(self.pitches)

        y, sr = sf.read(self.in_file)
        start = 0
        end = 0
        count = len(sorted(os.listdir(self.phrase_dir))) + 1

        for i in range(len(self.pitches)-1):
            if self.pitches[i] == 0 and self.pitches[i+1] >= 50.0:
                start = i
            if self.pitches[i] >= 50.0 and self.pitches[i+1] == 0:
                end = i+1
                # print(start, end)
                if int(int((end/10)*sr)-int((start/10)*sr)) > int(22050/2):
                    phrase = y[int((start/10)*sr):int((end/10*sr))]
                    # phrase = phrase / np.max(np.abs(phrase))
                    sf.write(os.path.join(self.phrase_dir, '{}.wav'.format(count)), samplerate=sr, data=phrase)
                    count+=1
                start = 0
                end = 0
        print('chunked audio by phrase for playback later')

    def normalize_phrases(self):

        for wavfile in os.listdir(self.phrase_dir):
            y, sr = sf.read(os.path.join(self.phrase_dir, wavfile))
            y = y / np.max(np.abs(y))
            sf.write(os.path.join(self.phrase_dir, wavfile), samplerate=sr, data=y)


            # rawsound = AudioSegment.from_file(wavfile, "wav")
            # normalizedsound = effects.normalize(rawsound)
            # normalizedsound.export(os.path.join(self.phrase_dir, wavfile), format="wav")
        print('normalized phrases')

    def truncate_silence(self):
        startPos = 0
        thresh = 15 # needs to be high bc audio has been normalized
        snd_array = []
        slnt_array = []
        sr, array = wav.read(self.in_file)
        chunk_len = int(sr/5)
        endPos=int(sr/5)

        try:

            for i in range(0, (len(array)-chunk_len), chunk_len): # I know this is horrible but the func still runs fast
                if np.mean(np.abs(array[startPos:endPos])) > thresh:
                    snd_array.append(array[startPos:endPos])
                if np.mean(np.abs(array[startPos:endPos])) < thresh:
                    slnt_array.append(array[startPos:endPos])
                startPos+=chunk_len
                endPos+=chunk_len

            snd_array = np.concatenate(snd_array).ravel()
            wav.write(self.in_file, sr, snd_array)
            print('removed silences')

            slnt_array = np.concatenate(slnt_array).ravel()
            wav.write(self.silent_file, sr, slnt_array)
            print('wrote silence file')

        except ValueError as e:
            pass

    def data_augmentation(self): # data augmentation through pitch shift
        tfm1 = sox.Transformer()
        tfm2 = sox.Transformer()
        tfm1.pitch(1.0)
        tfm1.build_file(self.in_file, os.path.join(self.aug_dir, 'aug1.wav'))
        tfm2.pitch(-1.0)
        tfm2.build_file(self.in_file, os.path.join(self.aug_dir, 'aug2.wav'))

        combined = []

        for file in os.listdir(self.aug_dir):
            sr, sound = wav.read(os.path.join(self.aug_dir, file))
            combined.append(sound)
        sr, sound2 = wav.read(self.in_file)
        combined.append(sound2)

        combined=np.hstack(combined)

        wav.write(self.in_file, rate=sr, data=combined.astype(np.int16)) # not sure why i converted to int16 here
        print('augmented data')

    def chunk_train_audio(self):
        chunk_len = 32768
        startpos = 0
        endpos = 32767
        count=0
        sr, data = wav.read(self.in_file)
        for i, n in enumerate(data):
            if i % chunk_len == 0:
                if len(data) - startpos >= 32768:
                    count+=1
                    chunk = data[startpos:endpos]
                    chunk = chunk/np.max(np.abs(chunk))
                    wav.write(os.path.join(self.chunk_dir,'{}.wav'.format(str(count).zfill(6))), sr, chunk)
                    startpos = (startpos+chunk_len-1)
                    endpos = (endpos+(chunk_len-1))
                else:
                    break

        print('chunked audio')

    def compute_CQTs(self):
        spec_count = 0
        for wavfile in sorted(os.listdir(self.chunk_dir)):
            sr, y = wav.read(os.path.join(self.chunk_dir, wavfile))
            y = y.astype(np.float64)
            # res_type='polyphase' should speed this up
            cqt = np.abs(librosa.cqt(y,  sr=sr, hop_length=512, fmin=64, n_bins=64, bins_per_octave=12, sparsity=0.01, res_type='polyphase'))
            cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
            spec_count+=1
            io.imsave(os.path.join(self.cqt_dir, '{}.jpg'.format(str(spec_count).zfill(6))), cqt_db)
        print('computed spectros')

    def get_CQT_layer(self):

        sr, y = wav.read(self.in_file)
        y = torch.FloatTensor(y)
        self.cqt = features.CQT(hop_length=512, fmin=64, n_bins=64, bins_per_octave=12)
        return self.cqt

    def compute_CQTs_GPU(self):

        for chunk in os.listdir(self.chunk_dir):
            sr, y = wav.read(os.path.join(self.chunk_dir, chunk))
            y = torch.FloatTensor(y)
            cqt_spec = self.cqt(y)
            cqt_spec = torch.abs(cqt_spec)
            cqt_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(cqt_spec)
            cqt_spec = cqt_spec.cpu().detach().numpy()
            # cqt_spec = cqt_spec.reshape((64, 64))
            cqt_spec = cqt_spec.reshape((cqt_spec.shape[1], cqt_spec.shape[2]))
            io.imsave(os.path.join(self.cqt_dir, chunk[:-4]+'.jpg'), cqt_spec)

    def cleanup(self):

        rmtree(self.aug_dir)
        rmtree(self.chunk_dir)
        #os.remove(self.in_file)
