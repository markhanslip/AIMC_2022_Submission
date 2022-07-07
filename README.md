# AIMC_2022_Submission

Code base for music work submission to AIMC 2022

Tested on Ubuntu 20.04 

This code is quite specific to my PhD project. It was intended more for my own usage, and for others to hack at and steal from. But should you feel compelled to run it in full, I'd suggest firstly creating an Anaconda environment for the dependencies. 

Install Anaconda: https://docs.anaconda.com/anaconda/install/

then run:
```
conda create -n [environment name] python=3.8 
conda activate [environment name] 
```
clone this repo: 
```
git clone https://github.com/markhanslip/AIMC_2022_Submission
cd AIMC_2022_Submission
```
install dependencies:
```
pip install -r requirements.txt
```
Once you're all set up, the main.py script runs the program. It takes 3 audio files - two of your playing, one just of the sound of your environment. These are processed into datasets, two classifiers are trained, and you then enter an interactive loop. 
```
python main.py --solo1 path/to/wavfile.wav --solo2 path/to/another/wavfile.wav --silence path/to/a/third/wavfile.wav 
```
