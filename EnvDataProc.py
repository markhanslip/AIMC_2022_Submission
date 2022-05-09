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

class EnvDataProcessing:

    def __init__(self, name: str, in_file: str): # change to accept a category name only
        self.classname = name
        self.in_file = in_file
        self.aug_dir = './{}_aug'.format(self.classname)
        self.train_dir = './env_train_data'
        self.chunk_dir = './{}_chunks'.format(self.classname)
        self.spec_dir = os.path.join(self.train_dir, './{}'.format(self.classname))
        self.phrase_dir = './{}_phrases'.format(self.classname)
        self.sr=22050
        self.mel = None
        self.pitches = None

    def make_dirs(self):
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
        if not os.path.exists(self.chunk_dir):
            os.mkdir(self.chunk_dir)
        if not os.path.exists(self.aug_dir):
            os.mkdir(self.aug_dir)
        if not os.path.exists(self.spec_dir):
            os.mkdir(self.spec_dir)
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
                    wav.write(os.path.join(self.chunk_dir,'{}.wav'.format(str(count).zfill(6))), sr, data[startpos:endpos])
                    startpos = (startpos+chunk_len-1)
                    endpos = (endpos+(chunk_len-1))
                else:
                    break

        print('chunked audio')

    def compute_mel_specs(self):
        spec_count = 0
        for wavfile in sorted(os.listdir(self.chunk_dir)):
            sr, y = wav.read(os.path.join(self.chunk_dir, wavfile))
            y = y.astype(np.float64)
            # res_type='polyphase' should speed this up
            mel = librosa.feature.melspectrogram(y,  sr=sr, hop_length=512, n_mels=64)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            spec_count+=1
            io.imsave(os.path.join(self.spec_dir, '{}.jpg'.format(str(spec_count).zfill(6))), mel_db)
        print('computed spectros')

    def get_mel_layer(self):

        sr, y = wav.read(self.in_file)
        y = torch.FloatTensor(y)
        self.mel = features.MelSpectrogram(hop_length=512, n_mels=64)
        return self.mel

    def compute_mel_specs_GPU(self):

        for chunk in os.listdir(self.chunk_dir):
            sr, y = wav.read(os.path.join(self.chunk_dir, chunk))
            y = torch.FloatTensor(y)
            mel_spec = self.mel(y)
            # cqt_spec = torch.abs(cqt_spec)
            mel_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
            mel_spec = mel_spec.cpu().detach().numpy()
            # cqt_spec = cqt_spec.reshape((64, 64))
            mel_spec = mel_spec.reshape((mel_spec.shape[1], mel_spec.shape[2]))
            io.imsave(os.path.join(self.spec_dir, chunk[:-4]+'.jpg'), mel_spec)

    def cleanup(self):

        rmtree(self.aug_dir)
        rmtree(self.chunk_dir)
        # os.remove(self.in_file)
