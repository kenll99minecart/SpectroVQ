from ..SpectrumProcessing import SpectraLoading
from .preprocessing import rescaleSpectra
import torch
import numpy as np

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
    
    def getReconstructSpectrum(self,addedIndices:np.ndarray = None):
        """
        Decode vector quantized indices back to mass spectra.

        Returns:
            tuple: (mz_arrays, intensity_arrays) for each decoded spectrum
        """
        with torch.no_grad():
            if not isinstance(self.indicesArr, torch.Tensor):
                self.indicesArr = torch.from_numpy(self.indicesArr)
            spectrum = self.model.reconstructIndices(self.indicesArr.to(self.device))
            
            if addedIndices is not None:
                if not isinstance(addedIndices, torch.Tensor):
                    addedIndices = torch.from_numpy(addedIndices)
                Originalspectrum = torch.square(rescaleSpectra(spectrum,dim = 2))
                Originalspectrum = torch.where(Originalspectrum < 1e-3,torch.zeros_like(Originalspectrum),Originalspectrum)
                temp_spectra_o = self.model.reconstructIndices(addedIndices.to(self.device))
                temp_spectra_o = rescaleSpectra(temp_spectra_o,dim = 2)
                temp_spectra_o = torch.square(temp_spectra_o)
                temp_spectra = torch.where(temp_spectra_o < 5e-3,torch.zeros_like(temp_spectra_o),temp_spectra_o)
                temp_spectra = torch.where(((temp_spectra > 1e-2) & (temp_spectra < 5e-1)),temp_spectra*3,temp_spectra)
                spectrum = torch.clip(Originalspectrum*1.0 + temp_spectra*0.4,0,1)
            else:
                spectrum = rescaleSpectra(spectrum,dim = 2)
                spectrum = torch.square(spectrum)
                spectrum = torch.where(spectrum < 1e-3,torch.zeros_like(spectrum),spectrum)
        return self.postprocessingSpectrum(spectrum.cpu().squeeze(1).numpy())
    
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
            output_intensity = list(output_intensity.astype(np.float32))
            output_mz = list(np.round(output_mz.astype(np.float32)+ 0.055 ,6))
            if len(self.leftmzList) > 0:
                leftmzArr = np.array(self.leftmzList[j])
                if (leftmzArr[leftmzArr >= 1500].shape[0] > 0) & (any([mz == 1499.95 for mz in output_mz])):
                    #print('removing peaks at 1499.95')
                    output_intensity.pop(output_mz.index(1499.95))
                    output_mz.pop(output_mz.index(1499.95))
                output_mz.extend(list(leftmzArr))
                output_intensity.extend(self.leftintensityList[j])
            sortedidx = np.argsort(output_mz)
            output_mz = np.array(output_mz,np.float32)[sortedidx]
            output_intensity = np.array(output_intensity,np.float32)[sortedidx]
            thresholdix = output_intensity > np.max(output_intensity)*0.01 # Clear small mz values for 1 percent mz
            output_mz,output_intensity = output_mz[thresholdix],output_intensity[thresholdix]
            
            output_mzList.append(output_mz)
            output_intensityList.append(output_intensity)
        return output_mzList, output_intensityList