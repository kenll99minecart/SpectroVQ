from ..SpectrumProcessing import SpectraLoading
import torch
import numpy as np

class SpectrumBatchEncoder():

    def __init__(self, mzList, intensityList,model):
        self.mzList = mzList
        self.intensityList = intensityList
        self.model = model
        self.device = next(self.model.parameters()).device

    def __len__(self):
        return len(self.mzList)
    
    def formBatch(self,**kwargs):
        BatchList = []
        MaxValList = []
        for i in range(len(self.mzList)):
            MSspectra,MaxVal = SpectraLoading.massSpectrumToVector(self.mzList[i],self.intensityList[i], bin_size = 0.1,SPECTRA_DIMENSION=13500,rawIT = False,Mode = 'sqrt',mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                           ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500])
            BatchList.append(torch.unsqueeze(MSspectra,dim = 0))
            MaxValList.append(MaxVal)
        self.Batch =  torch.unsqueeze(torch.vstack(BatchList),dim = 1)
        self.MaxValList = MaxValList

    def getReconstructSpectra(self):
        self.formBatch()
        with torch.no_grad():
            output_spectra, _, _ = self.model(self.Batch.to(self.device))
            return output_spectra
    
    def getReconstructIndices(self,outputOutOfRange = False,quantizer = 6):
        self.formBatch()
        with torch.no_grad():
            _, codes, _, _ = self.model.encode(self.Batch.to(self.device),returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
            if outputOutOfRange:
                output_mz, output_intensity = [],[]
                for i in range(len(self.mzList)):
                    original_mz, original_intensity = np.array(self.mzList[i]),np.array(self.intensityList[i])
                    missingidx = (original_mz < 150) | (original_mz >= 1500)
                    output_mz.append(original_mz[missingidx])
                    output_intensity.append(original_intensity[missingidx])
                return codes,output_mz,output_intensity
        return codes


class SpectrumBatchDecoder():

    def __init__(self, indicesList,model, leftovermzList = [], leftoverintensityList = [],MaxValList = []):
        '''
        indicesList: List of indices for the codebook; len(indicesList)  = Length '''
        self.leftmzList = leftovermzList
        self.leftintensityList = leftoverintensityList
        self.IndicesList = indicesList
        self.model = model
        self.device = next(self.model.parameters()).device
        self.MaxValList = MaxValList if len(MaxValList) > 0 else [1]*len(indicesList)

    def __len__(self):
        return len(self.IndicesList)
    
    def getReconstructSpectra(self):
        self.formBatch()
        with torch.no_grad():
            output_spectra, _, _ = self.model(self.Batch.to(self.device))
            return output_spectra
    
    def posprocessingSpectrum(self,spectrum):
        output_mzList,output_intensityList = [],[]
        for j in range(spectrum.shape[0]):
            output_mz,output_intensity = SpectraLoading.VectorToMassSpectrum(spectrum[j,:],self.MaxValList[j],bin_size = 0.1,threshold = 1e-4,min_mz = 150,AlterMZ=False,returnNumpy=True)#1e-3
            output_intensity = list(output_intensity**2)
            output_mz = list(np.round(output_mz + 0.05,6))
            if self.leftmzList:
                if (self.leftmzList[self.leftmzList >= 1500].shape[0] > 0) & (any([mz == 1499.95 for mz in output_mz])):
                    #print('removing peaks at 1499.95')
                    output_intensity.pop(output_mz.index(1499.95))
                    output_mz.pop(output_mz.index(1499.95))
                output_mz.extend(self.leftmzList)
                output_intensity.extend(self.leftintensityList)
            sortedidx = np.argsort(output_mz)
            output_mz = np.array(output_mz)[sortedidx]
            output_intensity = np.array(output_intensity)[sortedidx]

            thresholdix = output_intensity > np.max(output_intensity)*0.01#
            output_mz,output_intensity = output_mz[thresholdix],output_intensity[thresholdix]
            
            output_mzList.append(output_mz)
            output_intensityList.append(output_intensity)
        return output_mz, output_intensity
    
    def getReconstructSpectrum(self):
        with torch.no_grad():
            if isinstance(self.IndicesList):
                self.IndicesList = torch.tensor(self.IndicesList)
            spectrum = self.model.decodeLatent(self.model.decodeIndices(self.IndicesList.to(self.device)))
        return self.posprocessingSpectrum(spectrum.cpu().numpy())