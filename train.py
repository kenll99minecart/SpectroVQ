import torch
import pandas as pd
import os
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from utils.SpectrumProcessing.SpectraLoading import massSpectrumToVector
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch import seed_everything
import numpy as np
import argparse
from utils.model import modelStream
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description = 'Training script for SpectroVQ')
parser.add_argument('--data_path', type=str, default='library.feather',help = 'data path to the consensus library file (.feather)')
parser.add_argument('--output_dir', type=str, default='.',help = 'output directory for the trained model')
parser.add_argument('--batch_size', type=int, default=32,help = 'batch size for training')
parser.add_argument('--num_workers', type=int, default=4,help = 'number of workers for data loading')

# Model hyperparameters
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of training epochs')

args = parser.parse_args()
output_dir = args.output_dir
data_path = args.data_path

seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")

def find0arr(arr):
    return len(arr['intensity'])

def PreprocessData():
    print('Loading Training Data ....')
    df_train = pd.read_feather(data_path)
    df_train = df_train.rename(columns={'Name':'CompoundName','Annotation':'Peak Comments'})
    df_train['Charge'] = df_train['CompoundName'].str.extract(r'(\d+)$').astype(int)

    df_train.rename(columns={'IT': 'intensity','MZ':'m/z',}, inplace=True)

    df_train['Peptide'] = df_train['CompoundName'].apply(lambda x: x.split('/')[0])
    df_train['Peptide'] = df_train['Peptide'].str.replace(r'\[[^\]]*\]', 'X', regex=True)
    df_train['Peptide'] = df_train['Peptide'].str.replace(r'[^A-Za-z]', '', regex=True)
    
    df_train['length'] = df_train.apply(find0arr,axis = 1)
    df_train = df_train[df_train['length'] > 10]
    df_train.drop('length',inplace = True,axis = 1)
    df_train['isHLA'] = df_train['Comment'].str.contains('_HLA_').astype(int)
    
    df_train['PeptideLen'] = df_train['Peptide'].apply(lambda x: len(x) > 10)
    from sklearn.model_selection import train_test_split
    df_train['class'] = df_train.apply(lambda x: str(x['PeptideLen']) + str(x['Charge']), axis=1)
    train_data, test_data = train_test_split(df_train, test_size=0.05, random_state=42)

    # Print the shapes of the training and testing sets
    print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)

    train_data.reset_index(drop = True, inplace= True)

    return train_data, test_data

def transformToTensor(x):
    return torch.from_numpy(np.array(x.to_list(),dtype='float32'))

Bin_size = 0.1
class SpectrumAugmentation():
    def __init__(self,seed = 10,returntype = 'torch'):
        self.randomizer = np.random.default_rng(seed=seed)
        if returntype == 'torch':
            self.returntype = torch.from_numpy
        elif returntype == 'numpy':
            self.returntype = None
        else:
            raise NotImplementedError('returntype must be either torch or numpy')

    def MovingAverage(self,x, w):
        '''
        x: input signal
        w: kernel size
        '''
        return np.convolve(x, np.ones(w), 'valid') / w

    def AddGausianNoise(self,x):
        '''
        x: input signal
        '''
        return x + self.randomizer.normal(0, 0.001, size=x.shape)

    def ModifyMZ(self,mz):
        MZModified = self.randomizer.binomial(len(mz),0.25)
        Changeidx = self.randomizer.integers(low= 0, high = len(mz), size = MZModified)
        for i in Changeidx:
            mz[i] = mz[i] + self.randomizer.normal(0,0.01)
        return mz

    def ApplyMovingAverage(self,x):
        MovingAverage = self.randomizer.binomial(len(x),0.5)
        Changeidx = self.randomizer.integers(low= 0, high = len(x)-11, size = MovingAverage)
        for u in Changeidx:
            if (x[u] == 0) & (x[u+1] == 0):
                continue
            else:
                x[u:u+11] = self.MovingAverage(x[u:u+11],w = 11)
        return x
    
    def AddPeriodicPeaks(self,x):
        NoisePeakWavelength = self.randomizer.random(x.shape[0])
        magnitude = self.randomizer.gamma(1,0.001, x.shape[0])
        temp = np.linspace(0,10,x.shape[0])
        periodicPeaks = (np.sin(2*np.pi*NoisePeakWavelength*temp)+1)*magnitude
        return x + periodicPeaks

    def AddRandomPeaks(self,x,lambdaPoisson = 1):
        # Generate simulated electronic noises
        NumberNoisePeaks = self.randomizer.integers(1,150)
        Position = self.randomizer.integers(low= 0, high = x.shape[-1], size = NumberNoisePeaks)
        # Generating Poisson-distributed random variables
        intensity = np.random.poisson(lam=lambdaPoisson, size=NumberNoisePeaks)
        intensity = intensity/100
        for i in range(NumberNoisePeaks):
            x[Position[i]] += intensity[i]
        return x

randomAugmentizer = np.random.default_rng(seed=42)
class SpectrumBatch(Dataset):
    def __init__(self, df):
        df.sort_values(by = 'PrecursorMZ',inplace = True)
        df.reset_index(drop = True,inplace = True)
        self.df = df
        self.Xmz = df['m/z']
        self.Xintensity = df['intensity']
        self.peptide = df['Peptide']
        self.PrecursorMz = df['PrecursorMZ']
        self.PrecursorMz = self.PrecursorMz.apply(lambda x: float(x))
        self.length = df.shape[0]
        self.base_seed = 42
        self.Augmentation = SpectrumAugmentation(seed = self.base_seed,returntype = 'torch')
        self.applyAugmentationRNG = np.random.default_rng(seed=self.base_seed)

    def __len__(self):
        return self.length
    
    def reset_rng(self):
        self.applyAugmentationRNG = np.random.default_rng(seed=self.base_seed + randomAugmentizer.integers(0,1000))
        self.Augmentation = SpectrumAugmentation(seed = self.base_seed + randomAugmentizer.integers(0,1000),returntype = 'torch')
    
    def set_epoch(self):
        self.reset_rng()
    
    def __getitem__(self, idx):
        mz,it = self.Xmz.iloc[idx],self.Xintensity.iloc[idx]
        precursormz = self.PrecursorMz.iloc[idx]
        # Peak Masking
        delete_mask = self.applyAugmentationRNG.binomial(1,0.1,size = len(mz))
        mz = mz[delete_mask == 0]
        it = it[delete_mask == 0]
        spectra, _ = massSpectrumToVector(self.Xmz.iloc[idx],self.Xintensity.iloc[idx], bin_size = Bin_size,SPECTRA_DIMENSION=13500,rawIT = False,Mode = None,mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                           ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500])
        augmentedSpectra,_ = massSpectrumToVector(mz,it, bin_size = Bin_size,SPECTRA_DIMENSION=13500,rawIT = False,Mode = None,mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                           ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500],returntorch=False)
        # Electronic Noises
        if self.applyAugmentationRNG.binomial(1,0.5):
            augmentedSpectra = self.Augmentation.AddRandomPeaks(augmentedSpectra,lambdaPoisson = 5)
        # Peptide Noises
        if self.applyAugmentationRNG.binomial(1,0.5):
            currentPrecursorMz = precursormz
            mixupMz = self.applyAugmentationRNG.uniform(currentPrecursorMz - 5,currentPrecursorMz + 5)# 5 Th Tolerance
            mixupidx = np.argmin(np.abs(self.PrecursorMz - mixupMz))
            N = 0
            while mixupidx == idx:
                mixupMz = self.applyAugmentationRNG.uniform(currentPrecursorMz - 5,currentPrecursorMz + 5)# 5 Th Tolerance
                mixupidx = np.argmin(np.abs(self.PrecursorMz - mixupMz))
                N += 1
                if N > 10:
                    break
            mixupidx = np.clip(mixupidx,a_min=0,a_max=self.length-1)
            mixupSpectra,_ = massSpectrumToVector(self.Xmz.iloc[mixupidx],self.Xintensity.iloc[mixupidx], bin_size = Bin_size,SPECTRA_DIMENSION=13500,rawIT = False,Mode = None,mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                           ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500],returntorch=False)
            mix = self.applyAugmentationRNG.uniform(0.1,0.5,size = 1)
            augmentedSpectra = np.clip((augmentedSpectra* (1-mix) + mixupSpectra*mix),a_min=0,a_max=1)
        return torch.unsqueeze(torch.sqrt(spectra),dim = 0), torch.unsqueeze(torch.sqrt(torch.from_numpy(augmentedSpectra)),0).float()

eps = 1e-6
cos2 = nn.CosineSimilarity(dim=2, eps=1e-6)
def SpectralConstrastLoss(output,target):
    Similarity = cos2(output,target)
    dotProduct = (2*torch.acos(Similarity))/(math.pi)
    return dotProduct.mean()

def rescaleSepectra(Spectra,dim):
    SpectraMax = torch.max(Spectra,dim = dim,keepdim=True)[0]
    return Spectra / SpectraMax

def spectral_entropy(spectrum,dim = 2,reweigh  = False):
    if reweigh:
        spectrum = torch.where(spectrum >= 3, torch.tensor(1.0), 0.25 + spectrum * torch.log(spectrum + eps))
    return -torch.sum(spectrum * torch.log(spectrum + eps), dim = dim)

# https://www.nature.com/articles/s41592-021-01331-z#Sec9
def spectral_entropy_loss(input, target,dim = 2):
    inputspectrum = input / (torch.sum(input,dim = dim,keepdim = True) + eps)
    targetspectrum = target / (torch.sum(target,dim = dim,keepdim = True) + eps)
    SpectrumAB = inputspectrum + targetspectrum
    SpectrumAB = SpectrumAB / 2 #(torch.sum(SpectrumAB,dim = dim,keepdim = True) + eps)
    spectral_entropyAB = spectral_entropy(SpectrumAB,dim = dim)
    spectral_entropyA = spectral_entropy(inputspectrum,dim = dim)
    spectral_entropyB = spectral_entropy(targetspectrum,dim = dim)
    return 1-(((2 * spectral_entropyAB) - spectral_entropyA - spectral_entropyB) / np.log(4))

Threshold = 1e-2
def peak_portion(input, target,dim = 2):
    inputspectrum = torch.clamp(input / Threshold,min = 0,max = 1)
    numinputpeaks = torch.sum(inputspectrum,dim = dim,keepdim=True)
    targetspectrum = torch.clamp(target / Threshold,min = 0,max = 1)
    numtargetpeaks = torch.sum(targetspectrum,dim = dim,keepdim=True)
    mismatch = torch.abs(inputspectrum - targetspectrum)
    noiseRatio = torch.sum(mismatch,dim = dim) / (numinputpeaks + numtargetpeaks + eps)
    return noiseRatio

weightCS, weightSE = 1.0, 1.0
class LightningModelSpectraStream(L.LightningModule):
    def __init__(self, model,learning_rate,len_train_loader,len_test_loader):
        super().__init__()
        self.model = model
        self.len_train_loader = len_train_loader
        self.len_test_loader = len_test_loader
        self.learning_rate = learning_rate
        self.AverageTestLossList= []
        self.AverageTrainingLoss = []
        self.rng = np.random.default_rng(42)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        spectra, aug_spectra = batch
        vq = bool(self.rng.integers(0,2))
        output, quantize_loss, _ = self.model(aug_spectra, use_vq=vq)
        OrigCosineLoss = SpectralConstrastLoss(output, spectra)
        
        QuantizeLoss = quantize_loss.mean()
        SE_Loss = spectral_entropy_loss(output, spectra, dim=2).mean()

        loss = weightSE * SE_Loss + weightCS * OrigCosineLoss + QuantizeLoss 

        if batch_idx % 5000 == 0:
            print(".", end="")
            print('Training Loss: {}'.format(loss.item()))
            print('SpectralConstrastiveLoss: {}'.format(OrigCosineLoss.item()))
            print('QuantizeLoss: {}'.format(QuantizeLoss.item()))
            print('SELoss: {}'.format(SE_Loss.item()))

        if torch.isnan(loss):
            print('NAN Loss Detected at Training batch {}'.format(batch_idx))
        else:
            self.AverageTrainingLoss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):

        spectra, aug_spectra = batch

        output, _, _ = self.model(spectra)

        OrigCosineLoss = SpectralConstrastLoss(output, spectra)

        SE_Loss = spectral_entropy_loss(output, spectra, dim=2).mean()


        loss = OrigCosineLoss + SE_Loss

        if batch_idx % 100 == 0:
            print('Testing Loss: {}'.format(loss.item()))
            print('SpectralConstrastiveLoss: {}'.format(OrigCosineLoss.item()))
            print('SELoss: {}'.format(SE_Loss.item()))

        if torch.isnan(loss):
            print('NAN Loss Detected at Testing batch {}'.format(batch_idx))
        else:
            self.AverageTestLossList.append(loss.item())
        self.log("val_loss", loss, sync_dist=True, on_epoch=True)

    def on_train_epoch_end(self):
        self.AverageTrainingLoss.clear()
        self.trainer.train_dataloader.dataset.set_epoch()
        self.rng = np.random.default_rng(42 + self.rng.integers(0,1000))
        return 
    
    def on_validation_epoch_end(self):
        AverageTLoss = np.mean(self.AverageTestLossList)
        self.log("val_mean_loss", AverageTLoss,sync_dist = True)
        self.AverageTestLossList.clear()
        self.trainer.val_dataloaders.dataset.set_epoch()
        return 
    
    def configure_optimizers(self):
        warmup_epochs = 7
        total_epochs = 50
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        num_train_optimization_steps = total_epochs * self.len_train_loader
        num_warmup_steps = warmup_epochs * self.len_train_loader
        lr_scheduler = {'scheduler': get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return {"optimizer": optimizer,"lr_scheduler": lr_scheduler}
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark=True 
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

print('Starting Training Program')


model_kwargs2 = {'inputChannel': 1, 'n_q':12, 'n_filters': 32,'n_residual_layers': 1, 'lstm': 0, 'ratios':[9,7,4,2] , 'final_activation': 'Sigmoid',
            'residual_kernel_size': 6,'codebook_size':1024, 'FiLMEncoder' : False, 'FiLMDecoder' :False,'flattenQuantization' : False,
            'quantizationdim': 256, 'biLSTM': False,'causal':False,'useQINCo':False,'inputSize' : 13500,'dimension':256,'kernel_size':8, 
            'encoders':False,'bucketTransform':False,'applyquantizerdroput':True,'transformer':6}
num_epochs = 50
SpectraStream = modelStream.SpectroVQ(**model_kwargs2)
# %%
learning_rate = 1e-4
batch_size = 32
torch.cuda.empty_cache()
# %%
print('Preprocessing Training Data')
traindata,testdata = PreprocessData()
# %%
print('Creating Training Data Loaders')
spectraTrainData = SpectrumBatch(traindata)
train_loader = DataLoader(spectraTrainData, batch_size=batch_size, shuffle=True,num_workers=28,drop_last=True,pin_memory=True, prefetch_factor = 24)
spectraTestData = SpectrumBatch(testdata)
test_loader = DataLoader(spectraTestData, batch_size=batch_size, shuffle=False,num_workers=28,drop_last=True,pin_memory = True, prefetch_factor = 24)
LightningTrainer = LightningModelSpectraStream(SpectraStream,learning_rate=learning_rate,len_train_loader = len(train_loader),len_test_loader = len(test_loader))
checkpoint_callback = ModelCheckpoint(dirpath=output_dir,monitor = 'val_mean_loss',filename='SpectroVQ-{epoch:02d}-{val_loss:.2f}',save_top_k=3,mode='min') 
early_stop_callback = EarlyStopping(monitor="val_mean_loss", min_delta=0.005, patience=4, verbose=True, mode="min",check_on_train_epoch_end = False)
trainer = L.Trainer(max_epochs = num_epochs, log_every_n_steps = 1, check_val_every_n_epoch = 3, callbacks=[early_stop_callback,checkpoint_callback],accelerator="gpu", devices=1, num_nodes=1, strategy="auto",gradient_clip_val=1)
trainer.fit(LightningTrainer,train_dataloaders = train_loader,val_dataloaders = test_loader)

# %%
