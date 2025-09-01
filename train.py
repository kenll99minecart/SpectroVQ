# %%
import torch
import pandas as pd
import os
import numpy as np
import math
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from SpectraLoading import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L
from lightning.pytorch import seed_everything

seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")
from pyteomics import mass
import numpy as np

def fragments(peptide, types=('b', 'y'), maxcharge=1):
    """
    The function generates all possible m/z for fragments of types
    `types` and of charges from 1 to `maxharge`.
    """
    for i in range(1, len(peptide)):
        for ion_type in types:
            for charge in range(1, maxcharge+1):
                if ion_type[0] in 'abc':
                    finalpeptide = peptide[:i]
                else:
                    finalpeptide = peptide[i:]
                if not finalpeptide[0].isalpha():
                    continue
                
                if ('+' in finalpeptide)| ('-' in finalpeptide):
                    rawpepsequence = "".join([aa for aa in list(finalpeptide) if aa.isalpha()])
                    numericalpepsequence = "".join([aa for aa in list(finalpeptide) if not aa.isalpha()])
                    plusminus = [aa for aa in list(finalpeptide) if aa in '+-']
                    numericalpepsequence = numericalpepsequence.replace('-','+-').replace('+','+-').split('+-')
                    numericalpepsequence = [float(aa) for aa in numericalpepsequence if len(aa) > 0]
                    finalmass = 0
                    for idx,element in enumerate(numericalpepsequence):
                        if plusminus[idx] == '+':
                            finalmass += float(element)
                        else:
                            finalmass -= float(element)
                    finalmass += mass.fast_mass(rawpepsequence, ion_type=ion_type, charge=charge)
                    yield finalmass, 1 - int(ion_type == 'b')* 0.5
                else:
                    try:
                        yield mass.fast_mass(
                            finalpeptide, ion_type=ion_type, charge=charge), 1 - int(ion_type == 'b')* 0.5
                    except:
                        import traceback
                        traceback.print_exc()
                        TypeError('Error in mass calculation')
                        print(finalpeptide)
    
def find0arr(arr):
    return len(arr['intensity'])

def generateTheoreticalSpectrum(peptide,charge = 1):
    mz = []
    intensity = []
    for mzi, intensityi in fragments(peptide, maxcharge=charge):
        mz.append(mzi)
        intensity.append(intensityi)
    return mz,intensity

def PreprocessData():
    # df_train = pd.read_feather('/data/data/NIST/MSP.feather')
    # df_train['PrecursorMZ'] = df_train['DataBase'].apply(lambda x: float(str(x).split('Parent=')[-1].split(' ')[0]))
    # df_train['Charge'] = df_train['DataBase'].apply(lambda x: int(str(x).split('Charge=')[-1].split(' ')[0]))
    # df_train = df_train.sample(frac=0.5)
    df_train = pd.read_feather('/data2/PeptideAtlas/library.feather')
    df_train = df_train.rename(columns={'Name':'CompoundName','Annotation':'Peak Comments'})
    df_train['Charge'] = df_train['CompoundName'].str.extract(r'(\d+)$').astype(int)
    df_train['AnnotatedArray'] = df_train['Peak Comments'].apply(lambda x: ['?' ==  peak[0] for peak in x])
    # df_train = pd.concat([df_train,df_train2],axis = 0)
    
    # df_train3 = pd.read_feather('/data/data/MassIVEKBV2/LIBRARY_AUGMENT-3cac0386-download_filtered_mgf_library-main.feather')
    # df_train3.rename(columns={'it':'IT','mz':'MZ','Mass':'DataBase'},inplace = True)
    # df_train3['Comment'] = 'MassIVEKBV2'
    # df_train = pd.concat([df_train,df_train3],axis = 0)

    df_train.rename(columns={'IT': 'intensity','MZ':'m/z',}, inplace=True)
    #df_train[['m/z','intensity']] = df_train.apply(lambda x: RemovePrecursorPeak(x['m/z'],x['intensity'],x['PrecursorMZ']),axis = 1)
    print(df_train.shape)
    df_train.columns

    df_train['Peptide'] = df_train['CompoundName'].apply(lambda x: x.split('/')[0])
    df_train['length'] = df_train.apply(find0arr,axis = 1)
    df_train = df_train[df_train['length'] > 10]
    df_train.drop('length',inplace = True,axis = 1)
    df_train['isHLA'] = df_train['Comment'].str.contains('_HLA_').astype(int)

    df_train['PrecursorMz'] = df_train['Comment'].apply(lambda x: x.split('AvePrecursorMz=')[-1].split(' ')[0])
    # df_train['AnnotatedPeaks'] = df_train['Peak Comments'].apply(lambda x: GetAnnotatedPeaks(eval(x))[0])
    # df_train = df_train[df_train['AnnotatedPeaks'] > 2]
    # df_train.drop('AnnotatedPeaks',inplace = True,axis = 1)
    #df_train = df_train.sample(100)
    from sklearn.model_selection import train_test_split

    train_data, test_data = train_test_split(df_train, test_size=0.05, random_state=42,stratify=df_train['isHLA'])

    # Print the shapes of the training and testing sets
    print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)

    train_data.reset_index(drop = True, inplace= True)

    return train_data, test_data

def transformToTensor(x):
    return torch.from_numpy(np.array(x.to_list(),dtype='float32'))

Bin_size = 0.1
Max_seq = 128
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
    
# %%
randomAugmentizer = np.random.default_rng(seed=42)
class SpectrumBatch(Dataset):
    def __init__(self, df):
        df.sort_values(by = 'PrecursorMz',inplace = True)
        df.reset_index(drop = True,inplace = True)
        self.Xmz = df['m/z']
        self.Xintensity = df['intensity']
        self.Annotation = df['AnnotatedArray']
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
        if self.applyAugmentationRNG.binomial(1,0.5):
            mz = self.Augmentation.ModifyMZ(mz)
        # indices_to_delete = []
        # for i in range(len(mz)):
        delete_mask = self.applyAugmentationRNG.binomial(1,0.1,size = len(mz))
        mz = mz[delete_mask == 0]
        it = it[delete_mask == 0]
        #         indices_to_delete.append(i)
        # mz = np.delete(mz, indices_to_delete)
        # it = np.delete(it, indices_to_delete)
        spectra, _ = massSpectrumToVector(self.Xmz.iloc[idx],self.Xintensity.iloc[idx], bin_size = Bin_size,SPECTRA_DIMENSION=13500,rawIT = False,Mode = 'sqrt',mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                           ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500])
        augmentedSpectra,_ = massSpectrumToVector(mz,it, bin_size = Bin_size,SPECTRA_DIMENSION=13500,rawIT = False,Mode = 'sqrt',mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                           ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500],returntorch=False)
        if self.applyAugmentationRNG.binomial(1,0.25):
            augmentedSpectra = np.clip(self.Augmentation.AddGausianNoise(augmentedSpectra),a_min=0,a_max=1)
        if self.applyAugmentationRNG.binomial(1,0.25):
            augmentedSpectra = self.Augmentation.ApplyMovingAverage(augmentedSpectra)
        if self.applyAugmentationRNG.binomial(1,0.25):
            mixupidx = np.clip(self.applyAugmentationRNG.integers(-20,20) + idx,a_min=0,a_max=self.length-1)
            mixupSpectra,_ = massSpectrumToVector(self.Xmz.iloc[mixupidx],self.Xintensity.iloc[mixupidx], bin_size = Bin_size,SPECTRA_DIMENSION=13500,rawIT = False,Mode = 'sqrt',mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                           ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500],returntorch=False)
            mix = self.applyAugmentationRNG.uniform(0.1,0.5,size = 1)
            augmentedSpectra = np.clip((augmentedSpectra* (1-mix) + mixupSpectra*mix),a_min=0,a_max=1)
        
        return torch.unsqueeze(spectra,dim = 0), torch.unsqueeze(torch.from_numpy(augmentedSpectra),0).float()
    
eps = 1e-6
cos2 = nn.CosineSimilarity(dim=2, eps=1e-6)
def SpectralConstrastLoss(output,target):
    Similarity = cos2(output,target)
    dotProduct = (2*torch.acos(Similarity))/(math.pi)
    return dotProduct.mean()

def rescaleSepectra(Spectra,dim):
    SpectraMax = torch.max(Spectra,dim = dim,keepdim=True)[0]
    return Spectra / SpectraMax

def spectral_entropy(spectrum,dim = 2,reweigh  = False):#1-
    #spectrum = spectrum / (torch.sum(spectrum,dim = dim,keepdim = True) + eps)
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
    # signal = 1- torch.abs(inputspectrum - targetspectrum)
    # signalRatio = torch.sum(signal,dim = dim) / inputspectrum.shape[dim]
    mismatch = torch.abs(inputspectrum - targetspectrum)
    noiseRatio = torch.sum(mismatch,dim = dim) / (numinputpeaks + numtargetpeaks + eps)# inputspectrum.shape[dim]
    return noiseRatio

learning_rate = 1e-5
#3e-5#3e-6#1e-5 
betas = (0.5,0.9)
criterion = nn.CrossEntropyLoss()

import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

weightCS, weightSE = 1.0, 1.0
class LightningModelSpectraStream(L.LightningModule):
    def __init__(self, model,learning_rate,len_train_loader,len_test_loader):
        super().__init__()
        self.model = model
        # for para in self.discriminator.parameters():
        #     para.requires_grad = False
        self.len_train_loader = len_train_loader
        self.len_test_loader = len_test_loader
        self.learning_rate = learning_rate
        #optimizer = self.configure_optimizers()
        self.AverageTestLossList= []
        self.AverageTrainingLoss = []
        self.rng = np.random.default_rng(42)
        #self.scheduler = lr_scheduler(optimizer)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        spectra,aug_spectra = batch
        vq = bool(self.rng.integers(0,2))
        output,quantize_loss,_ = self.model(aug_spectra,vq)
        OrigCosineLoss = SpectralConstrastLoss(output,spectra)
        QuantizeLoss =   quantize_loss.mean()
        #FocalLoss = FocalLossFunc(output,spectra)
        SE_Loss = spectral_entropy_loss(output,spectra,dim = 2).mean()
        # peak_loss = peak_portion(output,spectra,dim = 2).mean()
        loss =  weightSE * SE_Loss + weightCS * OrigCosineLoss  + quantize_loss.mean()#+  weightCS* peak_loss
        if batch_idx % 5000 == 0:
            print(".", end = "")
            wandb.log({"Training Loss": loss.item(),'SpectralConstrastiveLoss':OrigCosineLoss.item(),'QuantizeLoss':QuantizeLoss.item(),'SELoss':SE_Loss.item(),
                       })#'PeakLoss':peak_loss.item()
        if torch.isnan(loss):
            print('NAN Loss Detected at Training batch {}'.format(batch_idx))
        else:
            self.AverageTrainingLoss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        #print(batch.shape)
        spectra,aug_spectra = batch
        #Condition = torch.ones(size = (spectra.shape[0],1),device = self.device,dtype = torch.float32)
        # Condition[:int(Condition.shape[0]/2)] = 0
        # ReverseCondition = torch.abs(1 - Condition)
        # outTest,quantize_lossTest = self.model(spectra,Condition)
        #ReverseCondition = torch.abs(1 - Condition)
        output,quantize_loss,_ = self.model(spectra)
        
        if batch_idx == 0:
            ValidationData = spectra[-1,:]
            aug_spectra = aug_spectra[-1,:]

            ValidationOutData, _, _  = self.model(ValidationData.unsqueeze(0))
            #print(ValidationOutData.shape)
            wandb.log({'PredictedVSOriginalSpectra':wandb.Image(plotComparingSpectra(ValidationData[0,:].cpu().numpy(),ValidationOutData[0,0,:].cpu().numpy(),'Original','Pred',num = 1350))})
            plt.close()
            wandb.log({'PredictedVSAugmentedSpectra':wandb.Image(plotComparingSpectra(aug_spectra[0,:].cpu().numpy(),ValidationOutData[0,0,:].cpu().numpy(),'Augmented','Pred',num = 1350))})
            plt.close()
            # mzAnnotated, itAnnotated = GetAnnotatedPeaks(str(detokenizePeptide(PeptideTokens[-1].cpu().numpy(),CodeBookSize=2048)),charge=charge[-1])
            # AnnotatedSpectra = massSpectrumToVector(np.array(mzAnnotated), np.array(itAnnotated), bin_size = Bin_size,SPECTRA_DIMENSION=20480,rawIT = False,Mode = None,mean0 = False,CenterIntegerBins=True,AlterMZ=True)[0]
            # wandb.log({'AnnotatedPeaks':wandb.Image(plotComparingSpectra(ValidationOutData[0,0,:].cpu().numpy(),AnnotatedSpectra,'Compressed','Annotated'))})
            # plt.close()


        OrigCosineLoss = SpectralConstrastLoss(output,spectra)#SpectralConstrastLoss(output,spectra)##*ReverseCondition
        QuantizeLoss = quantize_loss.mean()
        SE_Loss = spectral_entropy_loss(output,spectra,dim = 2).mean()

        loss = OrigCosineLoss + QuantizeLoss  + SE_Loss#peak_loss*(1-0.7977283363773258) #- 0.5*HoyerSparsity(BeforeQuantization).mean()*0.7977283363773258
        if batch_idx % 100 == 0:
            wandb.log({'SpectralConstrastiveTestLoss':OrigCosineLoss.item(),'QuantizeTestLoss':QuantizeLoss.item(),'SETestLoss':SE_Loss.item(),
                       })#'PeakTestLoss':peak_loss.item()

        if torch.isnan(loss):
            print('NAN Loss Detected at Testing batch {}'.format(batch_idx))
        else:
            self.AverageTestLossList.append(loss.item())
        self.log("val_loss", loss,sync_dist = True, on_epoch=True)

    def on_train_epoch_end(self):
        wandb.log({"Training Epoch Loss": np.average(self.AverageTrainingLoss)})
        self.AverageTrainingLoss.clear()
        self.trainer.train_dataloader.dataset.set_epoch()
        self.rng = np.random.default_rng(42 + self.rng.integers(0,1000))
        return 
    
    def on_validation_epoch_end(self):
        AverageTLoss = np.mean(self.AverageTestLossList)
        self.log("val_mean_loss", AverageTLoss,sync_dist = True)
        wandb.log({"Test Epoch Loss": AverageTLoss})
        self.AverageTestLossList.clear()
        self.trainer.val_dataloaders.dataset.set_epoch()
        return 
    
    def configure_optimizers(self):
        warmup_epochs = 7
        total_epochs = 50
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        # create learning rate scheduler
        num_train_optimization_steps = total_epochs * self.len_train_loader
        num_warmup_steps = warmup_epochs * self.len_train_loader
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

from torch.utils.data import DataLoader, Sampler

def main(train_data,test_data):
    spectraTrainData = SpectrumBatch(train_data)
    #print(spectraTrainData[0][1].shape)
    batch_size = 32
    train_loader = DataLoader(spectraTrainData, batch_size=batch_size, shuffle=True,num_workers=28,drop_last=True,pin_memory=True, prefetch_factor = 16)#,sampler=train_sampler
    #int(os.environ["SLURM_CPUS_PER_TASK"])
    # for u in train_loader:
    #     print(u.shape)
    #     break
    spectraTestData = SpectrumBatch(test_data)
    test_loader = DataLoader(spectraTestData, batch_size=batch_size, shuffle=False,num_workers=28,drop_last=True,pin_memory = True, prefetch_factor = 16)

    import importlib
    import modelStream
    importlib.reload(modelStream)
    model_kwargs2 = {'inputChannel': 1, 'n_q':16, 'n_filters': 32,'n_residual_layers': 1, 'lstm': 0, 'ratios':[9,7,6,2] , 'final_activation': 'Sigmoid',
                'residual_kernel_size': 6,'codebook_size':1024, 'FiLMEncoder' : False, 'FiLMDecoder' :False,'flattenQuantization' : False,
                'quantizationdim': 512, 'biLSTM': False,'causal':False,'useQINCo':False,'inputSize' : 13500,'dimension':512,'kernel_size':11, 
                'encoders':False,'bucketTransform':False,'applyquantizerdroput':True,'transformer':8}
    #SpectraStream = modelStream.VQMSStream(inputChannel = 1,n_q = 64,n_residual_layers = 3,lstm = 2,final_activation = 'Sigmoid',residual_kernel_size=10)#lstm = 0
    SpectraStream = modelStream.VQMSStream(**model_kwargs2)
    num_epochs = 50

    wandb.init(project="VQMSStreamConsensusMini",
        group = 'pytorchLightningNoCasual',
            config={
                "learning_rate":learning_rate,"optimizer":"AdamW","architecture": "VQMSMSStream(NoDiscriminator)", "Loss": "SpectralConstrastiveLoss,", "epochs:":num_epochs,'BinningSize':Bin_size,'Batch Size':batch_size,
                'save_path':'/home/james/ResidualVector/SpectraStreamNewOptimizedMini6ReproducedWithSchedulerWithSortPrecursorMZDifferentAugmentation/LightningChk',
                **model_kwargs2
            })

    # Magic
    log_freq = 50
    wandb.watch(SpectraStream, log_freq=log_freq)

    torch.cuda.empty_cache()
    import signal
    #
    #Training loop
    LightningTrainer = LightningModelSpectraStream(SpectraStream,learning_rate=learning_rate,len_train_loader = len(train_loader),len_test_loader = len(test_loader))
    checkpoint_callback = ModelCheckpoint(dirpath='/home/james/ResidualVector/SpectraStreamNewOptimizedMini6ReproducedWithSchedulerWithSortPrecursorMZDifferentAugmentation/LightningChk',monitor = 'val_mean_loss',filename='SpectraStream-{epoch:02d}-{val_loss:.2f}',save_top_k=3,mode='min') 
    early_stop_callback = EarlyStopping(monitor="val_mean_loss", min_delta=0.005, patience=4, verbose=True, mode="min",check_on_train_epoch_end = False)
    trainer = L.Trainer(max_epochs = num_epochs, log_every_n_steps = 1, check_val_every_n_epoch = 3, callbacks=[early_stop_callback,checkpoint_callback],accelerator="gpu", devices=1, num_nodes=1, strategy="auto",gradient_clip_val=1)#,plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)])#ddp_spawn
    trainer.fit(LightningTrainer,train_dataloaders = train_loader,val_dataloaders = test_loader)
    wandb.finish()
# %%
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark=True 
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    print('Starting Program')
    traindata,validationdata = PreprocessData()
    # %%
    main(traindata,validationdata)

# %%
