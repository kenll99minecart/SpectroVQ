import torch

def rescaleSpectra(Spectra,dim):
    SpectraMax = torch.max(Spectra,dim = dim,keepdim=True)[0]
    SpectraMin = torch.min(Spectra,dim = dim,keepdim=True)[0]
    return (Spectra - SpectraMin)/(SpectraMax - SpectraMin + 1e-6)