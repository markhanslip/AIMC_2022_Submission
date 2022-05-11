from EnvDataPreProc import EnvDataProcessing as EDP
from SoloDataPreProc import SoloDataProcessing as SDP
import CNN_AIMC as CNN
from PitchBasedOnsetDetector import OnsetsFilter as Filter
import time
import random
import os
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import numpy as np
import soundfile as sf
import argparse

def run_process(opt):

    ################## DATA PREPROCESSING ####################

    # combine both sax recs for env/sax discrimination:

    solo1 = AudioSegment.from_wav(opt.solo1)
    solo2 = AudioSegment.from_wav(opt.solo2)
    combined = solo1.append(solo2)
    combined.export('./combined.wav', format='wav')

    proc1 = EDP(name='playing', in_file='./combined.wav')
    proc1.make_dirs()
    proc1.resample_audio()
    proc1.data_augmentation()
    proc1.chunk_train_audio()
    proc1.get_mel_layer() # you can replace these two lines with proc1.compute_mel_specs() if no GPU
    proc1.compute_mel_specs_GPU()
    proc1.cleanup()

    proc2 = EDP(name='not_playing', in_file='./silence.wav')
    proc2.make_dirs()
    proc2.resample_audio()
    proc2.data_augmentation()
    proc2.chunk_train_audio()
    proc2.get_mel_layer() # you can replace these two lines with proc2.compute_mel_specs() if no GPU
    proc2.compute_mel_specs_GPU()
    proc2.cleanup()

    proc3 = SDP(name='solo1', in_file='./solo1.wav')
    proc3.make_dirs()
    proc3.resample_audio()
    proc3.chunk_by_phrase()
    proc3.truncate_silence()
    proc3.data_augmentation()
    proc3.chunk_train_audio()
    proc3.get_CQT_layer() # you can replace these two lines with proc3.compute_CQTs() if no GPU
    proc3.compute_CQTs_GPU()
    proc3.cleanup()

    proc4 = SDP(name='solo2', in_file='./solo2.wav')
    proc4.make_dirs()
    proc4.resample_audio()
    proc4.chunk_by_phrase()
    proc3.truncate_silence()
    proc4.data_augmentation()
    proc4.chunk_train_audio()
    proc4.get_CQT_layer() # you can replace these two lines with proc4.compute_CQTs() if no GPU
    proc4.compute_CQTs_GPU()
    proc4.cleanup()

    # ###################### MODEL TRAINING ###################

    envtrain = CNN.Trainer(data_path='./env_train_data/')
    envtrain.calculate_mean_std()
    envtrain.load_data()
    envtrain.build_model()
    envtrain.train(epochs=10)
    envtrain.save_model(model_path='./env_model.pth')

    saxtrain = CNN.Trainer(data_path='./solo_train_data/')
    saxtrain.calculate_mean_std()
    saxtrain.load_data()
    saxtrain.build_model()
    saxtrain.train(epochs=10)
    saxtrain.save_model(model_path='./solo_model.pth')

    ####################### LOAD MODELS FOR INFERENCE #########################

    # load classifiers: #

    infer_env = CNN.Inference(model_path = './env_model_exp.pth', rec_path = './infer.wav', spec_path = './infer_env.jpg', spec_type='mel')
    infer_env.load_model()

    infer_solo = CNN.Inference(model_path = './solo_model_exp.pth', rec_path = './infer.wav', spec_path = './infer_solo.jpg', spec_type='cqt')
    infer_solo.load_model()

    ###################### PLAYBACK FUNC ######################### this needs another layer before it tbh, it's too random

    def playback(predicted_class:str):

        snd1 = AudioSegment.from_file(os.path.join('./{}_phrases'.format(predicted_class), random.choice(os.listdir('./{}_phrases'.format(predicted_class)))))
        snd2 = AudioSegment.from_file(os.path.join('./{}_phrases'.format(predicted_class), random.choice(os.listdir('./{}_phrases'.format(predicted_class)))))
        combined = snd1.append(snd2, crossfade=50)
        combined = combined.fade_in(50).fade_out(50)
        _play_with_simpleaudio(combined)

    ############################ INTERACTIVE LOOP ###################################
    solo1_count=0
    solo2_count=0

    while True:

        rec = Recorder(channels=1)
        with rec.open('./infer.wav', 'wb') as recfile:
            recfile.record(duration=1.5)
        time.sleep(0.05)
        rec.resample_audio()
        time.sleep(0.05)
        rec.truncate_pad()
        time.sleep(0.05)

        infer_file = AudioSegment.from_wav('./infer.wav')
        infer_file = infer_file + 5.4 # amplify in same way that orig data was
        infer_file.export('./infer.wav', "wav")
        time.sleep(0.1)

        filter = Filter(in_file='./infer.wav', threshold=2)
        filter.get_freqs()
        filter.freqs_to_MIDI()
        result = filter.get_onsets()

        if result==0:
            print('no input detected')
            pass
        if result==1:
            infer_env.compute_spectro_GPU()
            env_predict, env_prob = infer_env.infer_class()
            if env_predict == 'playing':
                if env_prob >= 0.7:
                    infer_solo.compute_spectro_GPU()
                    solo_predict, solo_prob = infer_solo.infer_class()
                    if solo_predict == 'solo1':
                        if solo_prob > 0.9:
                            solo1_count += 1
                            solo2_count=0
                            if solo1_count >= 2:
                                if sax_prob >= 0.7:
                                    playback('solo1')
                                else:
                                    pass
                            else:
                                pass
                    if solo_predict == 'solo2':
                        if solo_prob > 0.9:
                            solo2_count += 1
                            solo1_count = 0
                            if solo_count >= 3:
                                if solo_prob >= 0.7:
                                    playback('solo2')
                                # t_count = 0
                                solo_count = 0
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
            if env_predict == 'silence':
                solo1_count = 0
                solo2_count = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solo1', type=str, help='path to recording of first solo in wav format')
    parser.add_argument('--solo2', type=str, help='path to recording of second solo in wav format')
    opt = parser.parse_args()
    run_process(opt)
