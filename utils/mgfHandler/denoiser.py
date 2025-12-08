from torch.utils.data import Dataset
from tqdm import tqdm
import os
from ..SpectrumProcessing import SpectraLoading
import torch
from pyteomics import mgf
import numpy as np
from ..model.modelStream import VQMSStream
import pandas as pd
eps = 1e-6

class SpectrumBatch(Dataset):
    def __init__(self, df):
        self.Xmz = df['m/z array']#
        self.Xintensity = df['intensity array']# array
        self.precursorMZ = df['PrecursorMZ'] if 'PrecursorMZ' in df.columns else None
        self.length = df.shape[0]

    def __len__(self):
        return self.length
    
    def getSpectrumVector(self,idx):
        return SpectraLoading.massSpectrumToVector(self.Xmz.iloc[idx],self.Xintensity.iloc[idx])

    def __getitem__(self, idx):
        MSspectra,MaxVal = SpectraLoading.massSpectrumToVector(self.Xmz.iloc[idx],self.Xintensity.iloc[idx], bin_size = 0.1,SPECTRA_DIMENSION=13500,rawIT = False,Mode = None,mean0 = False,CenterIntegerBins=False,AlterMZ=False
                                        ,GenerateMZList=False,MZDiff = False,mzRange = [150,1500])# 'sqrt'
        precursorpeakidx = np.floor((self.precursorMZ.iloc[idx] - 150) / 0.1).astype('int32')#'sqrt'
        return torch.sqrt(torch.unsqueeze(MSspectra,dim = 0)),MaxVal,precursorpeakidx

class MGFDenoiser():
    '''
    This class is reposible for the direct denoising operation from mgf to mgf files
    '''
    def __init__(self,model:VQMSStream,inputMGF,outputMGF):
        self.inputMGF = inputMGF
        if not os.path.exists(self.inputMGF):
            Warning(f'Input file {self.inputMGF} does not exist')
        self.outputMGF = outputMGF
        self.model = model
        self.device = next(self.model.parameters()).device
        
    def denoiseMGF(self,numQuantSingle = 6,retainOriginalPeaks = True,OnePercentThreshold = True,compoundedSpectra = True):
        with torch.no_grad():
            file = self.inputMGF
            reader = mgf.read(file)
            filename = os.path.splitext(os.path.basename(file))[0]
            ListofDicts = []
            for u in reader:
                ListofDicts.append(u)
            df_train = pd.DataFrame.from_dict(ListofDicts)
            df_train['length'] = df_train['m/z array'].apply(len)
            df_train['title'] = df_train['params'].apply(lambda x: x['title'] if 'title' in x else 'Unknown')
            print('Processing:',file)
            df_train['PrecursorMZ'] = df_train['params'].apply(lambda x: x['pepmass'][0])#/ int(x['charge'][0]))
            df_train['charge'] = df_train['params'].apply(lambda x: int(x['charge'][0]))
            
            spectraTrainData = SpectrumBatch(df_train)
            from torch.utils.data import DataLoader
            # Create a batch loader for the training data
            batch_size = 32
            train_loader = DataLoader(spectraTrainData, batch_size=batch_size, shuffle=False,num_workers=48,drop_last=False,pin_memory=True, prefetch_factor = 10)
            
            from tqdm import tqdm
            from matplotlib import pyplot as plt
            import importlib
            importlib.reload(SpectraLoading)
            def rescaleSepectra(Spectra,dim):
                SpectraMax = torch.max(Spectra,dim = dim,keepdim=True)[0]
                SpectraMin = torch.min(Spectra,dim = dim,keepdim=True)[0]
                return (Spectra - SpectraMin)/(SpectraMax - SpectraMin + 1e-6)

            rawIntensity = False
            addOriginalPeaks = retainOriginalPeaks
            compoundedSpectra = True
            output_intensityList, output_mzList = [], []
            savedChimericSpectra = []
            savedChimericIdx = []
            maxValList = []
            with torch.no_grad():
                for i, (train_data,MaxVal,precursorPeakIdx) in tqdm(enumerate(train_loader)):
                    input_spectra = train_data
                    input_spectra = input_spectra.to(self.device)

                    output_spectra, _,_ = self.model.forwardWithnumQuantizers(input_spectra,numQuantSingle)
                    
                    if not rawIntensity:
                        output_spectra = rescaleSepectra(output_spectra,dim = 2)
                        output_spectra = torch.square(output_spectra)
                        output_spectra = torch.where(output_spectra < 1e-3,torch.zeros_like(output_spectra),output_spectra)

                    # Check for compoundedSpectra spectrum
                    # Implement Chimeric Spectrum reconstruction
                    sqInput_spectra = torch.square(input_spectra)
                    if compoundedSpectra:
                        # Implement Chimeric Spectrum reconstruction
                        NumberOfPeaksPortion = torch.sum((output_spectra > 1e-1)&(sqInput_spectra > 1e-3),dim = 2) / torch.sum(sqInput_spectra > 1e-3,dim = 2)
                        residual_spectra = torch.where((sqInput_spectra > 1e-3)&(output_spectra < 1e-1),sqInput_spectra,0)
                        residual_spectra = torch.sqrt(rescaleSepectra(residual_spectra,dim = 2))
                        temp_spectra, _,_ = self.model.forwardWithnumQuantizers(residual_spectra,numQuantSingle)
                        temp_spectra_o = torch.square(rescaleSepectra(temp_spectra,dim = 2))
                        temp_spectra = torch.where(temp_spectra_o < 5e-3,torch.zeros_like(temp_spectra_o),temp_spectra_o)
                        temp_spectra = torch.where(((temp_spectra > 1e-2) & (temp_spectra < 5e-1)),temp_spectra*3,temp_spectra)
                        for u in range(NumberOfPeaksPortion.shape[0]):
                            NumPeaksSingle = NumberOfPeaksPortion[u].item()
                            # pass the spectrum if below threshold
                            if NumPeaksSingle < 0.5:
                                savedChimericSpectra.append(output_spectra[u,:].squeeze(0).squeeze(0).cpu().numpy().copy())
                                savedChimericIdx.append(i*batch_size+u)
                                maxValList.append(MaxVal[u])
                                output_spectra[u,:] = torch.clip(output_spectra[u,:]*1.0 + temp_spectra[u,:]*0.4,0,1)#0.2/0.4
                            

                    output_spectra = np.squeeze(output_spectra.cpu().numpy(),axis = 1)
                    input_spectra = np.squeeze(input_spectra.cpu().numpy(),axis = 1)
                  
                    for j in range(input_spectra.shape[0]):
                
                        if rawIntensity:
                            output_mz,output_intensity = SpectraLoading.VectorToMassSpectrum(output_spectra[j,:],1,bin_size = 0.1,threshold = 1e-4,min_mz = 150,AlterMZ=False,returnNumpy=True)
                            output_intensity = list(output_intensity)
                        else:
                            output_mz,output_intensity = SpectraLoading.VectorToMassSpectrum(output_spectra[j,:],MaxVal[j],bin_size = 0.1,threshold = 1e-4,min_mz = 150,AlterMZ=False,returnNumpy=True)
                            output_intensity = list(output_intensity)#**2
                        output_mz = list(np.round(output_mz+ 0.055 ,6))
                        if addOriginalPeaks:
                            original_mz, original_intensity = df_train.iloc[i*batch_size+j]['m/z array'], df_train.iloc[i*batch_size+j]['intensity array']
                            missingidx = (original_mz < 150) | (original_mz >= 1500)
                            if (original_mz[original_mz >= 1500].shape[0] > 0) & (any([mz == 1499.955 for mz in output_mz])):
                                #print('removing peaks at 1499.95')
                                output_intensity.pop(output_mz.index(1499.955))
                                output_mz.pop(output_mz.index(1499.955))
                            output_mz.extend(original_mz[missingidx])
                            output_intensity.extend(original_intensity[missingidx])
                        sortedidx = np.argsort(output_mz)
                        output_mz = np.array(output_mz,np.float32)[sortedidx]
                        output_intensity = np.array(output_intensity,np.float32)[sortedidx]
                        if output_intensity.shape[0]  == 0:
                            print(f'Empty at {j+i*128}')
                            output_mzList.append(output_mz)
                            output_intensityList.append(output_intensity)
                            continue
                        if OnePercentThreshold:
                            thresholdix = output_intensity > np.max(output_intensity)*0.01
                            output_mz,output_intensity = output_mz[thresholdix],output_intensity[thresholdix]
                        
                        output_mzList.append(output_mz)
                        output_intensityList.append(output_intensity)
                df_train['m/zNew'] = output_mzList
                df_train['intensityNew'] = output_intensityList
                
                if compoundedSpectra:
                    print('Start Processing Chimeric Spectrum')
                    chunk_size = 100  # Process 100 spectra at a time
                    df_chunks = []
                    
                    for chunk_start in range(0, len(savedChimericSpectra), chunk_size):
                        # print(f'processing chunk {chunk_start // chunk_size + 1}')
                        chunk_end = min(chunk_start + chunk_size, len(savedChimericSpectra))
                        RowList = []
                        
                        for i in range(chunk_start, chunk_end):
                            # if i % 100 == 0:
                            #     print(f"batch {i // 100 + 1}")
                            # print(maxValList[i])
                            output_mz,output_intensity = SpectraLoading.VectorToMassSpectrum(savedChimericSpectra[i],maxValList[i],bin_size = 0.1,threshold = 1e-4,min_mz = 150,AlterMZ=False,returnNumpy=True)
                            output_intensity = list(output_intensity)
                            output_mz = list(np.round(output_mz+ 0.055 ,6))
                            sortedidx = np.argsort(output_mz)
                            output_mz = np.array(output_mz,np.float32)[sortedidx]
                            output_intensity = np.array(output_intensity,np.float32)[sortedidx]
                            if output_intensity.shape[0] == 0:
                                print(f'Empty at {i}')
                                continue
                                
                            if OnePercentThreshold:
                                thresholdix = output_intensity > np.max(output_intensity)*0.01
                                output_mz = output_mz[thresholdix]
                                output_intensity = output_intensity[thresholdix]
                            
                            Newrow = df_train.iloc[savedChimericIdx[i]].copy()
                            # print(id(Newrow['params']),id(df_train.iloc[savedChimericIdx[i]]))
                            Newrow['m/zNew'] = output_mz
                            Newrow['intensityNew'] = output_intensity
                            NewDict = Newrow['params'].copy()
                            NewDict['title'] = Newrow['params']['title'] + '_Chimeric_' + str(i)
                            #Manually shift 0.01s rt to record compoundedSpectra time (hopefully don't affect much)  #658.98822
                            NewDict['rtinseconds'] += 0.03
                            Newrow['params'] = NewDict
                            if len(output_mz) < 2:
                                print(f'Empty at compoundedSpectra {i}')
                                continue
                            RowList.append(Newrow)
                            
                        if RowList:
                            df_chunks.append(pd.concat(RowList, axis=1).T)
                        del RowList
                    
                    if df_chunks:
                        df_chimeric = pd.concat(df_chunks).reset_index(drop = True)
                        print('Chimeric Shape')
                        print(df_chimeric.shape)
                        df_train = pd.concat([df_train, df_chimeric]).reset_index(drop = True)
                        del df_chunks, df_chimeric
              
                print('Done Conversion!')
                print(df_train[df_train['title'] == 'LFQ_Orbitrap_DDA_Human_03.35099.35099.2'])
                mzColumnName = 'm/zNew'#df_train.columns[df_train.columns.str.contains('m/z')][0]
                intensityColumnName = 'intensityNew'#df_train.columns[df_train.columns.str.contains('intensity')][0]

                from pyteomics import mgf
                #def writemzML(df,mzColumnName,intensityColumnName):
                def changeTofloat(arr):
                    return np.round(np.array(arr,np.float64),4)
                
                def writeMGF(df,mzColumnName,intensityColumnName,OutputPath):
                    if file.endswith('.mgf'):
                        Seq_mzs = df.apply(lambda x: {'m/z array':changeTofloat(list(x[mzColumnName])),'intensity array':np.array(list(x[intensityColumnName]),np.float64),'params':x['params']},axis = 1)#,'Charge':x['charge array'],'params':{'Title':x['title'],'File':x['filetitle'],'NativeID':x['ID'],'PEPMASS':str(x['PrecursorMasses']) + ' ' + str(x['PrecursorIT']),'RTINSECONDS':x['RT']*60,'Charge':str(x['Charge'])+'+','CollisionEnergy':x['CollisionEnergy']}},axis = 1)
                    elif file.endswith('.mzXML'):
                        Seq_mzs = df.apply(lambda x: {'m/z array':changeTofloat(list(x[mzColumnName])),'intensity array':np.array(list(x[intensityColumnName]),np.float64),'params':{'Title':str(filename) + '_'+x['id'],'File':str(filename),'NativeID':x['id'],'PEPMASS':str(x['PrecursorMZValue']) + ' ' + str(x['precursorMz'][0]['precursorIntensity']),'RTINSECONDS':x['retentionTime'],'Charge':str(x['charge'])+'+','CollisionEnergy':x['collisionEnergy']}},axis = 1)

                    def getSpectra(Seq_mzs):
                        for i, row in Seq_mzs.items():
                            yield {
                                'm/z array': row['m/z array'],
                                'intensity array': row['intensity array'],
                                'params': row['params']
                            }
                    mgf.write(getSpectra(Seq_mzs),OutputPath)

                writeMGF(df_train,mzColumnName,intensityColumnName,self.outputMGF)
                print(f'Wrote to {self.outputMGF}')
                print('Done!')
