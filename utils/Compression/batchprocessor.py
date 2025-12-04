from ..SpectrumProcessing import SpectraLoading
from .preprocessing import rescaleSpectra
import torch
import numpy as np

class SpectrumBatchEncoder():
    """
    Class for encoding batches of mass spectra using a neural network model.
    """

    def __init__(self, mzList, intensityList,model,chimeric = False):
        """
        Initialize the SpectrumBatchEncoder.

        Args:
            mzList (list): List of m/z arrays for each spectrum
            intensityList (list): List of intensity arrays for each spectrum
            model: Neural network model for encoding
            chimeric (bool): Whether to detect and handle chimeric spectra
        """
        self.mzList = mzList
        self.intensityList = intensityList
        self.model = model
        self.device = next(self.model.parameters()).device
        self.chimeric = chimeric
        self.chimericCodes = []
        self.chimericIdxList = []

    def __len__(self):
        """
        Return the number of spectra in the batch.

        Returns:
            int: Number of spectra
        """
        return len(self.mzList)
    
    def formBatch(self,**kwargs):
        """
        Convert m/z and intensity lists to a batch tensor for model input.

        Args:
            **kwargs: Additional keyword arguments (currently unused)
        """
        BatchList = []
        MaxValList = []
        for i in range(len(self.mzList)):
            MSspectra,MaxVal = SpectraLoading.massSpectrumToVector(self.mzList[i],self.intensityList[i], bin_size = 0.1,SPECTRA_DIMENSION=13500,rawIT = False,Mode = 'sqrt',mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                           ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500])
            BatchList.append(torch.unsqueeze(MSspectra,dim = 0))
            MaxValList.append(MaxVal)
        self.Batch =  torch.unsqueeze(torch.vstack(BatchList),dim = 1)
        self.MaxValList = MaxValList
    
    def getReconstructIndices(self,outputOutOfRange = False,quantizer = 6):
        """
        Encode spectra and return vector quantized indices.

        Args:
            outputOutOfRange (bool): Whether to return out-of-range peaks separately
            quantizer (int): Number of quantizers to use

        Returns:
            torch.Tensor or tuple: Encoded indices, optionally with chimeric data
        """
        self.formBatch()
        with torch.no_grad():
            inputBatch = self.Batch.to(self.device)
            if self.chimeric:
                OutputFullSpectra, _,_ = self.model.forwardWithnumQuantizers(inputBatch,16)
                OutputFullSpectra = rescaleSpectra(OutputFullSpectra,dim = 2)
                NumberOfPeaksPortion = torch.sum((OutputFullSpectra > 1e-2)&(inputBatch > 1e-2),dim = 2) / torch.sum(inputBatch > 1e-2,dim = 2)
                residual_spectra = torch.where((inputBatch > 1e-2)&(OutputFullSpectra < 1e-2),inputBatch,0)
                residual_spectra = rescaleSpectra(residual_spectra,dim = 2)
                _, ChimericCodes,_,_ = self.model.encode(residual_spectra,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
                for i in range(len(self.mzList)):
                    if NumberOfPeaksPortion[i] < 0.6:
                        self.chimericCodes.append(ChimericCodes[:,i,:])
                        self.chimericIdxList.append(i)
            _, codes, _, _ = self.model.encode(inputBatch,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
            if outputOutOfRange:
                output_mz, output_intensity = [],[]
                for i in range(len(self.mzList)):
                    original_mz, original_intensity = np.array(self.mzList[i]),np.array(self.intensityList[i])
                    missingidx = (original_mz < 150) | (original_mz >= 1500)
                    output_mz.append(original_mz[missingidx])
                    output_intensity.append(original_intensity[missingidx])
                return codes,output_mz,output_intensity
            if self.chimeric:
                return codes, self.chimericCodes, self.chimericIdxList
        return codes

class SpectrumBatchDecoder():
    """
    Class for decoding vector quantized indices back to mass spectra.
    """

    def __init__(self, indicesArr,model, leftovermzList = [], leftoverintensityList = [],MaxValList = []):
        """
        Initialize the SpectrumBatchDecoder.

        Args:
            indicesArr: Array of vector quantized indices
            model: Neural network model for decoding
            leftovermzList (list): List of leftover m/z values (optional)
            leftoverintensityList (list): List of leftover intensity values (optional)
            MaxValList (list): List of maximum values for each spectrum
        """
        '''
        indicesList: List of indices for the codebook; len(indicesList)  = Length '''
        self.leftmzList = leftovermzList
        self.leftintensityList = leftoverintensityList
        self.indicesArr = indicesArr
        self.model = model
        self.device = next(self.model.parameters()).device
        self.MaxValList = MaxValList if len(MaxValList) > 0 else [1]*indicesArr.shape[0]

    def __len__(self):
        """
        Return the number of spectra in the batch.

        Returns:
            int: Number of spectra
        """
        return self.indicesArr.shape[0]
    
    def getReconstructSpectrum(self):
        """
        Decode vector quantized indices back to mass spectra.

        Returns:
            tuple: (mz_arrays, intensity_arrays) for each decoded spectrum
        """
        with torch.no_grad():
            if not isinstance(self.indicesArr, torch.Tensor):
                self.indicesArr = torch.from_numpy(self.indicesArr)
            spectrum = self.model.reconstructIndices(self.indicesArr.to(self.device))
            # print(spectrum.shape)
        return self.postprocessingSpectrum(spectrum.cpu().squeeze().numpy())
    
    def postprocessingSpectrum(self,spectrum):
        """
        Post-process decoded spectra to convert back to m/z and intensity arrays.

        Args:
            spectrum: Decoded spectrum tensor

        Returns:
            tuple: (mz_arrays, intensity_arrays) for each processed spectrum
        """
        output_mzList,output_intensityList = [],[]
        for j in range(spectrum.shape[0]):
            output_mz,output_intensity = SpectraLoading.VectorToMassSpectrum(spectrum[j,:],self.MaxValList[j],bin_size = 0.1,threshold = 1e-4,min_mz = 150,AlterMZ=False,returnNumpy=True)#1e-3
            output_intensity = list(output_intensity.astype(np.float64)**2)
            output_mz = list(np.round(output_mz.astype(np.float64),6))# + 0.05
            if self.leftmzList:
                if (self.leftmzList[self.leftmzList >= 1500].shape[0] > 0) & (any([mz == 1499.95 for mz in output_mz])):
                    #print('removing peaks at 1499.95')
                    output_intensity.pop(output_mz.index(1499.95))
                    output_mz.pop(output_mz.index(1499.95))
                output_mz.extend(self.leftmzList)
                output_intensity.extend(self.leftintensityList)
            sortedidx = np.argsort(output_mz)
            output_mz = np.array(output_mz,np.float64)[sortedidx]
            output_intensity = np.array(output_intensity,np.float64)[sortedidx]
            thresholdix = output_intensity > np.max(output_intensity)*0.001 #Clear small mz values
            output_mz,output_intensity = output_mz[thresholdix],output_intensity[thresholdix]
            
            output_mzList.append(output_mz)
            output_intensityList.append(output_intensity)
        return output_mzList, output_intensityList