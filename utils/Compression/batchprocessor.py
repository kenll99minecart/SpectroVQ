from ..SpectrumProcessing import SpectraLoading
from .preprocessing import rescaleSpectra
import torch
import numpy as np

class SpectrumBatchEncoder():
    """
    Class for encoding batches of mass spectra using a neural network model.
    """

    def __init__(self, mzList, intensityList,model,compounded = False):
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
        self.compounded = compounded
        self.compoundedCodes = []
        self.compoundedIdxList = []

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
            MSspectra,MaxVal = SpectraLoading.massSpectrumToVector(self.mzList[i],self.intensityList[i], bin_size = 0.1,SPECTRA_DIMENSION=13500,rawIT = False,Mode = None,mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                        ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500])
            BatchList.append(torch.unsqueeze(torch.sqrt(MSspectra),dim = 0))
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
            sqinputBatch = torch.square(inputBatch)
            if self.compounded:
                # Process the output spectra
                output_spectra, _,_ = self.model.forwardWithnumQuantizers(inputBatch,quantizer)
                output_spectra = rescaleSpectra(output_spectra,dim = 2)
                output_spectra = torch.square(output_spectra)
                output_spectra = torch.where(output_spectra < 1e-3,torch.zeros_like(output_spectra),output_spectra)
                torch.save(output_spectra,'/home/james/SpectroVQ/test2.tensor')
                # Calculate the peaks portion
                NumberOfPeaksPortion = torch.sum((output_spectra > 1e-1)&(sqinputBatch > 1e-3),dim = 2) / torch.sum(inputBatch > 1e-3,dim = 2)
                residual_spectra = torch.where((sqinputBatch > 1e-3)&(output_spectra < 1e-1),sqinputBatch,0)
                residual_spectra = torch.sqrt(rescaleSpectra(residual_spectra,dim = 2))
                _, ChimericCodes,_,_ = self.model.encode(residual_spectra,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
                for i in range(len(self.mzList)):
                    if NumberOfPeaksPortion[i] < 0.5:
                        self.compoundedCodes.append(ChimericCodes[:,i,:])
                        self.compoundedIdxList.append(i)
            _, codes, _, _ = self.model.encode(inputBatch,returnCodebookIndices = True,returnAll = True,numquantizer = quantizer)
            if outputOutOfRange:
                output_mz, output_intensity = [],[]
                for i in range(len(self.mzList)):
                    original_mz, original_intensity = np.array(self.mzList[i]),np.array(self.intensityList[i])
                    missingidx = (original_mz < 150) | (original_mz >= 1500)
                    output_mz.append(original_mz[missingidx])
                    output_intensity.append(original_intensity[missingidx])
                if self.compounded:
                    return codes,self.compoundedCodes, self.compoundedIdxList,output_mz,output_intensity
                else:
                    return codes,output_mz,output_intensity
            if self.compounded:
                return codes, self.compoundedCodes, self.compoundedIdxList, 'dummy'
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
                Originalspectrum = self.model.reconstructIndices(addedIndices.to(self.device))
                Originalspectrum = rescaleSpectra(Originalspectrum,dim = 2)
                Originalspectrum = torch.square(Originalspectrum)
                Originalspectrum = torch.where(Originalspectrum < 1e-3,torch.zeros_like(Originalspectrum),Originalspectrum)
                temp_spectra_o = torch.square(rescaleSpectra(spectrum,dim = 2))
                temp_spectra = torch.where(temp_spectra_o < 5e-3,torch.zeros_like(temp_spectra_o),temp_spectra_o)
                temp_spectra = torch.where(((temp_spectra > 1e-2) & (temp_spectra < 5e-1)),temp_spectra*3,temp_spectra)
                spectrum = torch.clip(Originalspectrum*1.0 + temp_spectra*0.4,0,1)
                torch.save(Originalspectrum,'/home/james/SpectroVQ/test.tensor')
            else:
                spectrum = rescaleSpectra(spectrum,dim = 2)
                spectrum = torch.square(spectrum)
                spectrum = torch.where(spectrum < 1e-3,torch.zeros_like(spectrum),spectrum)
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
            output_intensity = list(output_intensity.astype(np.float32))
            output_mz = list(np.round(output_mz.astype(np.float32)+ 0.055 ,6))# + 0.05
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
            thresholdix = output_intensity > np.max(output_intensity)*0.01 #Clear small mz values for 1 percent mz
            output_mz,output_intensity = output_mz[thresholdix],output_intensity[thresholdix]
            
            output_mzList.append(output_mz)
            output_intensityList.append(output_intensity)
        return output_mzList, output_intensityList