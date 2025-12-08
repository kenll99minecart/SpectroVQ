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
        MSspectra,MaxVal = SpectraLoading.massSpectrumToVector(self.Xmz.iloc[idx],self.Xintensity.iloc[idx], bin_size = 0.1,SPECTRA_DIMENSION=13500,rawIT = False,Mode = None,mean0 = False,CenterIntegerBins=False,AlterMZ=False
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

    def getReconstructIndices(self,outputOutOfRange = False,quantizer = 6,batch_size = 128, num_workers = 16):
        DataLoaderBatch = DataLoader(self.dataset,batch_size = batch_size,shuffle = False,num_workers = num_workers,drop_last = False)
        CodeList = []
        with torch.no_grad():
            for batchidx, (inputBatch, MaxVal) in enumerate(DataLoaderBatch):
                self.MaxValList.extend(MaxVal.numpy().tolist())
                inputBatch = inputBatch.to(self.device)
                if self.compounded:
                    output_spectra, _,_ = self.model.forwardWithnumQuantizers(inputBatch,quantizer)
                    NumberOfPeaksPortion = torch.sum((output_spectra > 1e-1)&(inputBatch > 1e-3),dim = 2) / torch.sum(inputBatch > 1e-3,dim = 2)
                    residual_spectra = torch.where((inputBatch > 1e-3)&(output_spectra < 1e-1),inputBatch,0)
                    residual_spectra = torch.sqrt(rescaleSpectra(residual_spectra,dim = 2))
                    # temp_spectra, _,_ = SpectraStream.forwardWithnumQuantizers(residual_spectra,quantizer)
                    # temp_spectra_o = torch.square(rescaleSpectra(temp_spectra,dim = 2))
                    # temp_spectra = torch.where(temp_spectra_o < 5e-3,torch.zeros_like(temp_spectra_o),temp_spectra_o)
                    # temp_spectra = torch.where(((temp_spectra > 1e-2) & (temp_spectra < 5e-1)),temp_spectra*3,temp_spectra)
                    # for u in range(NumberOfPeaksPortion.shape[0]):
                    #     NumPeaksSingle = NumberOfPeaksPortion[u].item()
                    #     # pass the spectrum if below threshold
                    #     if NumPeaksSingle < 0.5:
                    #         output_spectra[u,:] = torch.clip(output_spectra[u,:]*1.0 + temp_spectra[u,:]*0.4,0,1)
                    _, ChimericCodes,_,_ = self.model.encode(residual_spectra,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
                    for i in range(residual_spectra.shape[0]):
                        if NumberOfPeaksPortion[i] < 0.5:
                            self.compoundedCodes.append(ChimericCodes[:,i,:])
                            self.compoundedCodesIdx.append(i + (batchidx * batch_size))
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
            if self.compounded:
                return finalCode, self.compoundedCodes, self.compoundedCodesIdx
        return finalCode