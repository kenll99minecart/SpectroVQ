from torch.utils.data import Dataset, DataLoader
from ..SpectrumProcessing import SpectraLoading
from .preprocessing import rescaleSpectra
import torch
import numpy as np

class SpectrumBatch(Dataset):
    def __init__(self, mzList,intensityList):
        self.Xmz = mzList
        self.Xintensity = intensityList
        self.length = len(mzList)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        MSspectra,MaxVal = SpectraLoading.massSpectrumToVector(self.Xmz[idx],self.Xintensity[idx], bin_size = 0.1,SPECTRA_DIMENSION=13500,rawIT = False,Mode = None,mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                        ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500])
        return torch.sqrt(torch.unsqueeze(MSspectra,dim = 0)),MaxVal


class SpectrumTorchBatchEncoder():
    '''
    This class is responsible to directly import All Spectrum from an mgf file
    '''
    def __init__(self, mzList, intensityList,model,compounded = False):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.dataset = SpectrumBatch(mzList,intensityList)
        self.compounded = compounded
        self.compoundedCodes = []
        self.compoundedCodesIdx = []
        self.MaxValList = []
        self.mzList, self.intensityList = mzList, intensityList
        
    def getReconstructIndices(self,outputOutOfRange = False,quantizer = 6,batch_size = 128, num_workers = 16):
        DataLoaderBatch = DataLoader(self.dataset,batch_size = batch_size,shuffle = False,num_workers = num_workers,drop_last = False)
        CodeList = []
        if outputOutOfRange:
            extendedmz,extendedit = [], []
        with torch.no_grad():
            for batchidx, (inputBatch, MaxVal) in enumerate(DataLoaderBatch):
                self.MaxValList.extend(MaxVal.numpy().tolist())
                inputBatch = inputBatch.to(self.device)
                sqinputBatch = torch.square(inputBatch)
                if self.compounded:
                    # Process the output spectra
                    output_spectra, _,_ = self.model.forwardWithnumQuantizers(inputBatch,quantizer)
                    output_spectra = rescaleSpectra(output_spectra,dim = 2)
                    output_spectra = torch.square(output_spectra)
                    output_spectra = torch.where(output_spectra < 1e-3,torch.zeros_like(output_spectra),output_spectra)
                    # Calculate the peaks portion
                    NumberOfPeaksPortion = torch.sum((output_spectra > 1e-1)&(sqinputBatch > 1e-3),dim = 2) / torch.sum(inputBatch > 1e-3,dim = 2)
                    residual_spectra = torch.where((sqinputBatch > 1e-3)&(output_spectra < 1e-1),sqinputBatch,0)
                    residual_spectra = torch.sqrt(rescaleSpectra(residual_spectra,dim = 2))
                    _, ChimericCodes,_,_ = self.model.encode(residual_spectra,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
                    for i in range(residual_spectra.shape[0]):
                        if NumberOfPeaksPortion[i] < 0.5:
                            self.compoundedCodes.append(ChimericCodes[:,i,:])
                            self.compoundedCodesIdx.append(i + (batchidx * batch_size))
                _, codes, _, _ = self.model.encode(inputBatch,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
                CodeList.append(codes.cpu())

                if outputOutOfRange:
                    for i in range(inputBatch.shape[0]):
                        original_mz, original_intensity = np.array(self.mzList[batchidx * batch_size + i]),np.array(self.intensityList[batchidx * batch_size + i])
                        missingidx = (original_mz < 150) | (original_mz >= 1500)
                        extendedmz.append(original_mz[missingidx])
                        extendedit.append(original_intensity[missingidx])
            finalCode = torch.concat(CodeList,dim = 1)
            
        if (self.compounded) & (outputOutOfRange):
            return codes,self.compoundedCodes, self.compoundedCodesIdx,extendedmz,extendedit
        else:
            if self.compounded:
                return codes,self.compoundedCodes, self.compoundedCodesIdx, 'dummy'
            if outputOutOfRange:
                return codes,extendedmz,extendedit
        return finalCode