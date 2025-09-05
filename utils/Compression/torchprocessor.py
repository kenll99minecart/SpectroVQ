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
        MSspectra,MaxVal = SpectraLoading.massSpectrumToVector(self.Xmz[idx],self.Xintensity[idx], bin_size = 0.1,SPECTRA_DIMENSION=13500,rawIT = False,Mode = 'sqrt',mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                        ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500])
        return torch.unsqueeze(MSspectra,dim = 0),MaxVal


class SpectrumTorchBatchEncoder():
    def __init__(self, mzList, intensityList,model,chimeric = False):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.dataset = SpectrumBatch(mzList,intensityList)
        self.chimeric = chimeric
        self.chimericCodes = []
        self.chimericIdxList = []
        self.MaxValList = []

    def getReconstructIndices(self,outputOutOfRange = False,quantizer = 6,batch_size = 128, num_workers = 16):
        DataLoaderBatch = DataLoader(self.dataset,batch_size = batch_size,shuffle = False,num_workers = num_workers,drop_last = False)
        CodeList = []
        with torch.no_grad():
            for batchidx, (inputBatch, MaxVal) in enumerate(DataLoaderBatch):
                self.MaxValList.extend(MaxVal.numpy().tolist())
                inputBatch = inputBatch.to(self.device)
                if self.chimeric:
                    OutputFullSpectra, _,_ = self.model.forwardWithnumQuantizers(inputBatch,16)
                    OutputFullSpectra = rescaleSpectra(OutputFullSpectra,dim = 2)
                    NumberOfPeaksPortion = torch.sum((OutputFullSpectra > 1e-2)&(inputBatch > 1e-2),dim = 2) / torch.sum(inputBatch > 1e-2,dim = 2)
                    residual_spectra = torch.where((inputBatch > 1e-2)&(OutputFullSpectra < 1e-2),inputBatch,0)
                    residual_spectra = rescaleSpectra(residual_spectra,dim = 2)
                    _, ChimericCodes,_,_ = self.model.encode(residual_spectra,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
                    for i in range(OutputFullSpectra.shape[0]):
                        if NumberOfPeaksPortion[i] < 0.6:
                            self.chimericCodes.append(ChimericCodes[:,i,:])
                            self.chimericIdxList.append(i + (batchidx * batch_size))
                _, codes, _, _ = self.model.encode(inputBatch,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
                CodeList.append(codes.cpu())

                # TODO: incoporate out of range handling
                if outputOutOfRange:
                    output_mz, output_intensity = [],[]
                    for i in range(len(self.mzList)):
                        original_mz, original_intensity = np.array(self.mzList[i]),np.array(self.intensityList[i])
                        missingidx = (original_mz < 150) | (original_mz >= 1500)
                        output_mz.append(original_mz[missingidx])
                        output_intensity.append(original_intensity[missingidx])
                    return codes,output_mz,output_intensity
                
                finalCode = torch.concat(CodeList,dim = 1)
            if self.chimeric:
                return finalCode, self.chimericCodes, self.chimericIdxList
        return finalCode