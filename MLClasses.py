import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.autograd import Variable
import os
import librosa
from PIL import Image
import imageio as io
from shutil import move
import torchaudio
from nnAudio import features
import soundfile as sf
from torchsummary import summary

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels = 3, out_channels = 32,
                      kernel_size = 3, stride=1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
            )

        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels = 32, out_channels = 64,
                      kernel_size = 3, stride=1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
            )

        self.fully_connected = nn.Sequential(

            nn.Linear(64*16*16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 2),
            )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        return nn.functional.softmax(x, dim=1)

class Trainer:

    def __init__(self, data_path):

        self.data_path = data_path
        self.mean = None
        self.std = None
        self.num_workers = os.cpu_count()
        self.classes = None
        self.train_data_loader = None
        self.valid_data_loader = None
        self.weights = None
        self.batch_size = 64
        self.input_res = 64
        self.device = None
        self.optimize = None
        self.loss_fn = None
        self.model = None

    def calculate_mean_std(self):
        # calculate the mean and std of dataset:
        temp_transform = transforms.ToTensor()
        temp_full_dataset = datasets.ImageFolder(root=self.data_path,
                                                 transform=temp_transform)
        temp_loader = torch.utils.data.DataLoader(temp_full_dataset,
                                                  shuffle=False,
                                                  num_workers=self.num_workers)

        dataset_means = []
        dataset_stds = []

        for i, data in enumerate(temp_loader, 0):

            sample = data[0].numpy()
            sample_mean = np.mean(sample, axis=(0,2,3))
            sample_std = np.std(sample, axis=(0,2,3), ddof=1) # ddof=1 consistent with torch.std
            dataset_means.append(sample_mean)
            dataset_stds.append(sample_std)

        self.mean = np.array(dataset_means).mean(axis=0)
        self.std = np.array(dataset_stds).mean(axis=0)
        self.mean = list(np.around(self.mean, decimals=3))
        self.std = list(np.around(self.std, decimals=3))

        print('mean = '+str(self.mean), 'std = '+str(self.std))

    def load_data(self):

        num_workers = os.cpu_count()
        batch_size = 64

        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_res),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.mean,
                                 std = self.std)
        ])

        full_dataset = datasets.ImageFolder(root=self.data_path,
                                            transform=transform)

        full_dataLoader = torch.utils.data.DataLoader(full_dataset,
                                                      shuffle=True,
                                                      num_workers=self.num_workers,
                                                      drop_last=True)

        train_size = int(0.95 * len(full_dataset))
        valid_size = len(full_dataset) - train_size

        train_data, valid_data = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

        self.train_data_loader = torch.utils.data.DataLoader(train_data,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             num_workers=self.num_workers,
                                                             drop_last=True)

        self.valid_data_loader = torch.utils.data.DataLoader(valid_data,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             num_workers=self.num_workers,
                                                             drop_last=True)

        self.classes=full_dataLoader.dataset.classes
        print('classes are '+str(self.classes))

    def build_model(self):

        self.model = CNN()

        for param in self.model.parameters():
            param.requires_grad=True

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("device is cuda")
        else:
            self.device = torch.device("cpu")
            print("device is cpu")
        self.model.to(self.device)
        print(self.model)
        summary(self.model, (3, 64, 64))

    def count_trainable_params(self):

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self, epochs):

        train_loss_per_epoch=[]
        valid_loss_per_epoch=[]

        for epoch in range(epochs):
            epoch += 1 # start at epoch 1 rather than 0
            training_loss = 0.0
            valid_loss = 0.0
            self.model.train()
            print('training epoch {}...'.format(epoch))

            for batch in self.train_data_loader:

                self.optimizer.zero_grad()

                inputs, labels = batch

                if self.device==torch.device("cuda"):
                    inputs = inputs.cuda()
                inputs = inputs.to(self.device)
                output = self.model(inputs)
                labels = labels.to(self.device)

                loss = self.loss_fn(output, labels)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.data.item()

            training_loss /= len(self.train_data_loader)
            print("epoch {} training done".format(epoch))

            self.model.eval()
            correct = 0
            num_correct = 0
            num_examples = 0

            print("about to run testing")

            for batch in self.valid_data_loader:
                inputs, labels = batch

                if self.device==torch.device("cuda"):
                    inputs = inputs.cuda()

                inputs = inputs.to(self.device)
                output = self.model(inputs)
                labels = labels.to(self.device)
                loss = self.loss_fn(output, labels)
                valid_loss += loss.data.item()
                correct = torch.eq(torch.max(nn.functional.softmax(output, dim=1), dim=1)[1], labels).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]

            try:
                valid_loss /= len(self.valid_data_loader)
            except ZeroDivisionError:
                pass

            train_loss_per_epoch.append(training_loss/len(self.train_data_loader))

            try:
                valid_loss_per_epoch.append(valid_loss/len(self.valid_data_loader))
            except ZeroDivisionError:
                pass

            try:
                print("epoch: {}, training loss: {:.3f}, validation loss: {:.3f}, accuracy = {:.2f}".format(epoch, training_loss, valid_loss, num_correct / num_examples))
            except ZeroDivisionError:
                pass

        print('Finished Training')

    def save_model(self, model_path):

        torch.save({
            'model':self.model.state_dict(),
            'input_res':self.input_res,
            'mean':self.mean,
            'std':self.std,
            'classes':self.classes,
            'device':self.device
        }, model_path)

class Inference: # think this should inherit from Trainer

    def __init__(self, model_path, rec_path, spec_path, spec_type):
        self.model_path = model_path
        self.rec_path = rec_path
        self.spec_path = spec_path
        self.spec_type = spec_type
        self.input_res = 64
        self.model = None
        self.mean = None
        self.std = None
        self.classes = None
        self.device = None
        self.device = None
        self.prediction = None
        self.probability = None
        self.cqt = None
        self.mel = None

    def load_model(self):

        model_data = torch.load(self.model_path, map_location='cpu')
        self.classes = model_data['classes']
        self.mean = model_data['mean']
        self.std = model_data['std']
        self.device = model_data['device']
        self.model = CNN()
        self.model.load_state_dict(model_data['model'])
        self.model.eval()
        self.model.to(self.device)
        print('loaded model')

    def compute_spectro(self):

        if self.spec_type == 'cqt':

            y, sr = sf.read(self.rec_path)
            if len(y) > 32768:
                y = y[:32767]
            y = y.astype(np.float64)
            cqt = np.abs(librosa.cqt(y,  sr=sr,
                                     hop_length=512, fmin=64,
                                     n_bins=64, bins_per_octave=12,
                                     sparsity=0.01, res_type='polyphase'))

            cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
            io.imsave(self.spec_path, cqt_db)
            print('computed spectro')

        elif self.spec_type == 'mel':

            y, sr = sf.read(self.rec_path)
            if len(y) > 32768:
                y = y[:32767]
            y = y.astype(np.float64)
            mel = librosa.feature.melspectrogram(y,  sr=sr,
                                                 hop_length=512, n_mels=64)

            mel_db = librosa.power_to_db(mel, ref=np.max)
            io.imsave(self.spec_path, cqt_db)
            print('computed spectro')

    def get_layer(self, in_file):

        if self.spec_type == 'cqt':
            y, sr = sf.read(in_file)
            y = torch.FloatTensor(y)
            self.cqt = features.CQT(hop_length=512, fmin=64,
                                    n_bins=64, bins_per_octave=12)
            return self.cqt

        elif self.spec_type == 'mel':
            y, sr = sf.read(in_file)
            y = torch.FloatTensor(y)
            self.mel = features.MelSpectrogram(hop_length=512, n_mels=64)
            return self.mel

    def compute_spectro_GPU(self):

        if self.spec_type == 'cqt':

            if self.cqt == None:

                y, sr = sf.read(self.rec_path)
                y = torch.FloatTensor(y)
                self.cqt = features.CQT(hop_length=512, fmin=64,
                                        n_bins=64, bins_per_octave=12)

            y, sr = sf.read(self.rec_path)
            if len(y)>32768:
                y = y[:32767]
            y = torch.FloatTensor(y)

            cqt_spec = self.cqt(y)
            cqt_spec = torch.abs(cqt_spec)
            cqt_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(cqt_spec)
            cqt_spec = cqt_spec.cpu().detach().numpy()
            cqt_spec = cqt_spec.reshape((cqt_spec.shape[1], cqt_spec.shape[2]))
            cqt_spec = Image.fromarray(cqt_spec)
            cqt_spec = cqt_spec.convert("RGB")
            io.imsave(self.spec_path, cqt_spec)

        elif self.spec_type == 'mel':

            if self.mel == None:

                y, sr = sf.read(self.rec_path)
                y = torch.FloatTensor(y)
                self.mel = features.MelSpectrogram(hop_length=512, n_mels=64)

            y, sr = sf.read(self.rec_path)
            if len(y)>32768:
                y = y[:32767]
            y = torch.FloatTensor(y)

            mel_spec = self.mel(y)
            mel_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
            mel_spec = mel_spec.cpu().detach().numpy()
            mel_spec = mel_spec.reshape((mel_spec.shape[1], mel_spec.shape[2]))
            mel_spec = Image.fromarray(mel_spec)
            io.imsave(self.spec_path, mel_spec)

    def infer_class(self):

        image = Image.open(self.spec_path)
        image = image.convert('RGB')
        image_tensor = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])(image)
        image_tensor = Variable(image_tensor, requires_grad=False)
        test_image = image_tensor.unsqueeze(0)

        if self.device == torch.device('cuda'):
            test_image = test_image.cuda()
        output = self.model(test_image)
        self.probability, self.prediction = torch.topk(output, len(self.classes))
        self.probability = self.probability[0].detach().cpu().numpy()
        self.prediction = self.prediction[0].cpu().numpy()
        print(self.prediction[0])

        self.prediction = self.classes[self.prediction[0]]
        self.probability = self.probability[0]
        print('prob:', self.probability)
        print('predicted class is {} with probability score {}'.format(self.prediction, self.probability))
        return self.prediction, self.probability
