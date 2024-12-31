import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from tqdm import tqdm

def normalizeMS(it):
    it = np.log(it)
    val = np.max(it)
    it = it/val
    return it

def massSpectrumToVector(mz_list,intensity_list,bin_size = 0.1,SPECTRA_DIMENSION = 20000, rawIT = True, mean0 = True, Mode = 'log',cap1 = True,returntorch = True,device = 'cpu',AlterMZ = True,CenterIntegerBins = True,GenerateMZList = False
                         ,MZDiff = False, mzRange:list = [None,None]):
    intensity_list = np.asarray(intensity_list,dtype = np.float32)
    mz_list = np.asarray(mz_list,dtype = np.float32)
    assert mz_list.shape[0] == intensity_list.shape[0]
    if not rawIT:
        if Mode == 'log':
            intensity_list = np.log(intensity_list+1)
            Minval = np.log(1)
        elif Mode == 'sqrt':
            intensity_list = np.sqrt(intensity_list)
            Minval = 0
        else:
            Minval = 0
        Maxval = np.max(intensity_list)
        # Minval = np.min(intensity_list)
        # Minval = np.clip(Minval,-np.inf,0)
        intensity_list = (intensity_list - Minval)/(Maxval - Minval)
    else:
        intensity_list = np.log(intensity_list)
        intensity_list = np.clip(intensity_list,0,np.inf)
    
    if mzRange[0] is not None:
        idx = mz_list >= mzRange[0]
        mz_list = mz_list[idx]
        intensity_list = intensity_list[idx]
        mz_list = mz_list - mzRange[0]
    if mzRange[1] is not None:
        idx = mz_list < mzRange[1]
        mz_list = mz_list[idx]
        intensity_list = intensity_list[idx]

    if GenerateMZList:
        OriginalMZ = mz_list
        if returntorch:
            mz_vector  = torch.zeros(SPECTRA_DIMENSION,device = device)
        else:
            mz_vector  = np.zeros(SPECTRA_DIMENSION, dtype='float32')

    if AlterMZ:
        mz_list = mz_list/1.00048
        OriginalMZ = mz_list
        mz_dummy_arr = np.arange(0,SPECTRA_DIMENSION*bin_size,bin_size)
    else:
        mz_dummy_arr = np.arange(0,SPECTRA_DIMENSION*bin_size,bin_size)
    
    assert mz_dummy_arr.shape[0] == SPECTRA_DIMENSION

    indexes = mz_list / bin_size

    if not CenterIntegerBins:
        indexes = np.floor(indexes).astype('int32')
    else:
        indexes = np.around(indexes).astype('int32')
   
    #SPECTRA_DIMENSION  = np.max(indexes)

    if returntorch:
        vector = torch.zeros(SPECTRA_DIMENSION, dtype = torch.float32,device = device)
    else:
        vector = np.zeros(SPECTRA_DIMENSION, dtype='float32')
    indexes = np.clip(indexes,0,SPECTRA_DIMENSION-1)
    #duplicated_indices = np.where(np.bincount(indexes) > 1)[0]
    if GenerateMZList:
        temp_it_list,temp_mz_list = [],[]
    #Same index, add up both intensity??
    for i, index in enumerate(indexes):
        vector[index] += intensity_list[i]
        if GenerateMZList:
            temp_it_list.append(intensity_list[i])
            temp_mz_list.append(OriginalMZ[i])
            if (i == (indexes.shape[0] - 1)):
                #end case
                assert len(temp_it_list) == len(temp_mz_list)
                #mz_vector[index] = np.sum([temp_it_list[u]*temp_mz_list[u] for u in range(len(temp_it_list))])/sumIT
                mz_vector[index] = temp_mz_list[np.argmax(temp_it_list)]
                if MZDiff:
                    mz_vector[index] = mz_vector[index]- mz_dummy_arr[index]# +0.05
                temp_it_list.clear()
                temp_mz_list.clear()
            elif (indexes[i+1] != index):#indexes[i]
                #sumIT = np.sum(temp_it_list)
                assert len(temp_it_list) == len(temp_mz_list)
                #mz_vector[index] = np.sum([temp_it_list[u]*temp_mz_list[u] for u in range(len(temp_it_list))])/sumIT
                mz_vector[index] = temp_mz_list[np.argmax(temp_it_list)]
                if MZDiff:
                    mz_vector[index] = mz_vector[index]- mz_dummy_arr[index] #+ 0.05
                temp_it_list.clear()
                temp_mz_list.clear()

    if not rawIT:
        if cap1:
            if returntorch:
                vector = torch.clip(vector,0,1)
            else:
                vector = np.clip(vector,0,1)

    if mean0:
        vector = 2*vector - 1
    if GenerateMZList:
        if returntorch:
            vector = torch.vstack([vector,mz_vector])
        else:
            vector = np.vstack([vector,mz_vector])
    return vector, Maxval

def VectorToMassSpectrum(Spec,MaxVal, bin_size = 0.01,threshold = 1e-7, min_mz = 0,AlterMZ = True,returnNumpy = True):
    if isinstance(Spec,torch.Tensor):
        Spec = Spec.detach().numpy()
    if isinstance(MaxVal,torch.Tensor):
        MaxVal = MaxVal.detach().numpy()
    
    nonzero_indices = np.where(Spec > threshold)[0]
    #print(nonzero_indices)
    intensity_list = Spec[nonzero_indices] * MaxVal
    #print(nonzero_indices)
    if AlterMZ:
        alterterm = 1.00048
    else:
        alterterm = 1
    mz_list = (nonzero_indices)*bin_size*alterterm + min_mz

    if returnNumpy:
        return np.round(mz_list,6),intensity_list
    else:
        mz_list = list(np.round(np.array(mz_list),6))#[round(mz,6) for mz in mz_list]#+ bin_size/2
        return mz_list,intensity_list

def CorrectMZ(mz,SPECTRA_DIMENSION,bin_size = 0.01,min_mz = 0,MZDiff = False,outputindices = False):
    mz_dummy_arr = np.arange(0,SPECTRA_DIMENSION*bin_size,bin_size)
    #nonzero_indices = np.where(mz > threshold)[0]
    #mz = mz[nonzero_indices]
    if MZDiff:
        mz = mz + mz_dummy_arr#[nonzero_indices]
    mz =  mz* 1.00048 + min_mz
    if outputindices:
        return mz#,nonzero_indices
    else:
        return mz

def CorrectIT(it,MaxVal,threshold = 1e-7):
    nonzero_indices = np.where(it > threshold)[0]
    it = it[nonzero_indices]
    it = it * MaxVal
    return it,nonzero_indices

def GetTopNPeaks(mz,intensity,peaknum = 10,returntorch = True, normalizeIntensity = False,mean0  = True):
    for dat in [mz,intensity]:
        if isinstance(dat,list):
            dat = np.array(dat)
        elif isinstance(dat,torch.Tensor):
            dat = dat.detach().numpy()
    if len(mz) < peaknum:
        peaknum = len(mz)
    if normalizeIntensity:
        intensity = np.log(intensity)
        if mean0:
            val = np.max(intensity)/2
            intensity = (intensity - val)/val 
        else:
            val = np.max(intensity)
            intensity = intensity/val

    indexes = np.argsort(intensity)[-peaknum:]
    #print(indexes)
    final_mz,final_it = np.array(mz,dtype = np.float32)[indexes], np.array(intensity,dtype = np.float32)[indexes]
    if returntorch:
        #print(final_mz)
        return torch.from_numpy(final_mz),torch.from_numpy(final_it)
    else:
        return final_mz,final_it
    
def FindPrecursorPeak(mz,intensity,precursorMZ,precursorTol = 0.1,returnIndices = True):
    if isinstance(mz,torch.Tensor):
        mz = mz.detach().numpy()
    if isinstance(intensity,torch.Tensor):
        intensity = intensity.detach().numpy()
    if isinstance(precursorMZ,torch.Tensor):
        precursorMZ = precursorMZ.detach().numpy()
    if isinstance(mz,list):
        mz = np.array(mz)
    if isinstance(intensity,list):
        intensity = np.array(intensity)
    if precursorTol is not None:
        idx = np.where((mz < (precursorMZ - precursorTol)) | (mz > (precursorMZ + precursorTol)))[0]
        if returnIndices:
            return idx
        else:
            return mz[idx],intensity[idx]
    else:
        #find the closest peak to precursorMZ
        idx = np.argmin(np.abs(mz - precursorMZ))
        if returnIndices:
            return idx
        else:
            return mz[idx],intensity[idx]

def GetFirstNPeaks(mz,intensity,peaknum = 10,returntorch = True, normalizeIntensity = False,mean0  = True,cutoffmz = 100):
    if isinstance(mz,torch.Tensor):
        mz = mz.detach().numpy()
    if isinstance(intensity,torch.Tensor):
        intensity = intensity.detach().numpy()
    if len(mz) < peaknum:
        peaknum = len(mz)
    if normalizeIntensity:
        intensity = np.log(intensity)
        if mean0:
            val = np.max(intensity)/2
            intensity = (intensity - val)/val 
        else:
            val = np.max(intensity)
            intensity = intensity/val

    indexes = np.argsort(mz)[:peaknum]
    final_mz,final_it = np.array(mz[indexes],dtype = np.float32), np.array(intensity[indexes],dtype = np.float32)
    if returntorch:
        return torch.from_numpy(final_mz),torch.from_numpy(final_it)
    else:
        return final_mz,final_it
    
def plotComparingSpectra(spectra1,spectra2,label1 = 'Spectra1',label2 = 'Spectra2',num = 2048,step = 0.1):
    u = [spectra1,spectra2]
    for i,spectra in enumerate(u):
        if spectra.min() != 0:
            u[i] = (spectra - np.min(spectra)) / (np.max(spectra) - np.min(spectra))
            
    assert u[0].shape == u[1].shape
    spectra1 = np.squeeze(u[0])
    spectra2 = np.squeeze(u[1])
    fig = plt.figure(figsize=(10,10))
    markerline1, stemlines1, baseline1 = plt.stem(np.arange(0,num,step = step),spectra1,label = label1,linefmt='b-',markerfmt=' ')
    plt.setp(stemlines1, 'linewidth', 2)
    markerline2, stemlines2, baseline2 = plt.stem(np.arange(0,num,step = step),-spectra2,label = label2,linefmt='r-',markerfmt=' ')
    plt.setp(stemlines2, 'linewidth', 2)
    plt.legend()
    return fig

def plotRawSpectra(mz1,intensity1,mz2,intensity2,label1 = 'original',label2 = 'predicted'):
    fig = plt.figure(figsize=(10,10))
    markerline1, stemlines1, baseline1 = plt.stem(mz1,intensity1,label = label1,linefmt='b-',markerfmt=' ')
    plt.setp(stemlines1, 'linewidth', 2)
    markerline2, stemlines2, baseline2 = plt.stem(mz2,-intensity2,label = label2,linefmt='r-',markerfmt=' ')
    plt.setp(stemlines2, 'linewidth', 2)
    plt.legend()
    return fig
    #print(pairwise.cosine_similarity(spectra1.reshape(1,-1),spectra2.reshape(1,-1)))

from pyteomics import pylab_aux as pa, usi
from pyteomics import mass
def fragmentsAA(peptide, types=('b', 'y'), maxcharge=1):
    """
    The function generates all possible m/z for fragments of types
    `types` and of charges from 1 to `maxcharge`.
    """
    for i in range(1, len(peptide)):
        for ion_type in types:
            for charge in range(1, maxcharge+1):
                if ion_type[0] in 'abc':
                    yield mass.fast_mass(
                            peptide[:i], ion_type=ion_type, charge=charge)
                else:
                    yield mass.fast_mass(
                            peptide[i:], ion_type=ion_type, charge=charge)

def GetAnnotatedPeaks(comments,charge = 2):
    '''
    Comments: can be str of dict
    dict: Peak Comments
    str: AA Sequence
    '''
    if isinstance(comments, dict):
        mzList = []
        for key,value in comments.items():
            if value[0] != '?':
                mzList.append(float(key))
    elif isinstance(comments, str):
        mzList = list(fragmentsAA(comments,maxcharge = charge))
        #raise NotImplementedError
    else:
        raise NotImplementedError
    itList = [1 for _ in range(len(mzList))]

    return mzList, itList

def EncodeSpectra(mz,it,delta_encoding = True,Max_length = 2048, intensityProcessing = [],mzProcessing = [], returntype = 'numpy'):
    assert len(mz) == len(it)
    if (len(mz) > Max_length/2) & (returntype != 'dual'):
        print('Spectra length is greater than Max_length, truncating')
        print(mz)
        print(it)
        mz = mz[:int(Max_length/2)]
        it = it[:int(Max_length/2)]
    elif (len(mz) > Max_length) & (returntype == 'dual'):
        print('Spectra length is greater than Max_length, truncating')
        print(mz)
        print(it)
        mz = mz[:Max_length]
        it = it[:Max_length]
    if ~np.all(np.diff(mz) >= 0):
        sortedIdx = np.argsort(mz)
        mz,it= map(lambda x: np.array(x)[sortedIdx], [mz,it])
    if delta_encoding:
        mz = np.concatenate([[mz[0]],np.diff(mz)],axis = 0)
    if len(intensityProcessing) > 0:
        for func in intensityProcessing:
            if func == 'log':
                it = np.log(it)
            elif func == 'sqrt':
                it = np.sqrt(it)
            elif func == 'MaxNorm':
                it = it/np.max(it)
            elif func == 'TICNorm':
                it = it/np.sum(it)
            elif func == 'CAP':
                it = np.clip(it,0,1)

    if len(mzProcessing) > 0:
        for func in mzProcessing:
            if func == 'MaxNorm':
                mz = (mz - np.min(mz))/(np.max(mz)-np.min(mz))
            elif func == 'CAP':
                mz = np.clip(mz,0,1)
            elif isinstance(func,tuple):
                if func[0] == 'Threshold':
                    mz = mz[mz > func[1]]
                    it = it[mz > func[1]]
                elif func[0] == 'divide':
                    if len(func)  == 2:
                        mz = mz / func[1]
                    elif len(func) == 3:
                        mz = (mz - func[1])/(func[2]-func[1])

    #originalLength = len(mz)
    if returntype == 'dual':
        mz =  np.pad(mz, (0, Max_length - len(mz)), 'constant')
        it =  np.pad(it, (0, Max_length - len(it)), 'constant')
        return mz,it
    elif returntype == 'numpy':
        vec = np.zeros(Max_length)
        mz =  np.pad(mz, (0, int(Max_length/2 - len(mz))), 'constant')
        it =  np.pad(it, (0, int(Max_length/2 - len(it))), 'constant')
        vec[::2] = mz
        vec[1::2] = it
    elif returntype == 'torch':
        vec = torch.zeros(Max_length)
        mz =  torch.tensor(np.pad(mz, (0, int(Max_length/2 - len(mz))), 'constant'),dtype=torch.float32)
        it =  torch.tensor(np.pad(it, (0, int(Max_length/2 - len(it))), 'constant'),dtype=torch.float32)
        vec[::2] = mz
        vec[1::2] = it
    return vec