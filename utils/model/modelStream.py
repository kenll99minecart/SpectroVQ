# This implementation is inspired from
# https://github.com/facebookresearch/encodec
# which is released under MIT License. Hereafter, the original license:
# MIT License
# %%
import math
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch import nn

from . import quantization as qt 
from . import modules as m

# Alternate implmentation of the modelStream class

class VQMSStream(nn.Module):
    def __init__(self, inputChannel = 1, dimension = 128, n_filters = 32,n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},final_activation: tp.Optional[str] = 'Tanh',
                 kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = True, norm: str = 'weight_norm',
                 lstm: int = 2,n_q : int = 16, codebook_size:int = 1024,kmeans_init: bool = True, kmeans_iters: int = 50, threshold_ema_dead_code: int = 2, FiLMEncoder = False, 
                 FiLMDecoder = False, flattenQuantization = False,quantizationdim = 128,biLSTM :bool = False,FixedCodeSize = None,inputSize = 20480,
                 encoders = False,bucketTransform = False, NumBucket = 4,quantl2 = False,applyquantizerdroput = True,transformer: int = 0):
        
        """StreamingModel.
    Args:
        SEANet Params:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method. Options: 'weight_norm','layer_norm','spectral_norm','time_group_norm'
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
        
        Codebook Params:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        flattenQuantization (bool): Whether to flatten the quantization output.
    """
        super().__init__()
        # if FiLM:
        #     self.FiLMlayers = nn.ModuleList()
        #     self.FiLMparams = FiLM_params
        #     if FiLM_params['Last_layer_only']:
        #         # for _ in range(2):
        #         #     self.FiLMlayers.append(nn.Linear(1,2))
        #         self.FiLMlayers.append(nn.Linear(1,2))
        #     else:
        #         raise NotImplementedError("FiLM is only implemented for the last layer")
        self.embeddingLength = math.ceil(inputSize / np.prod(ratios))

        self.encoder = m.SEANetEncoder(channels = inputChannel, dimension = dimension, n_filters = n_filters,n_residual_layers = n_residual_layers,
                 ratios = ratios, activation= activation, activation_params = activation_params, kernel_size=kernel_size,
                 last_kernel_size = last_kernel_size, residual_kernel_size = residual_kernel_size, dilation_base= dilation_base, causal=causal,
                 lstm = lstm,norm = norm,AffineTransform=FiLMEncoder,biLSTM = biLSTM,transformer = transformer,transformer_args={'max_seq_len':self.embeddingLength})
        
        self.decoder = m.SEANetDecoder(channels = inputChannel, dimension = dimension, n_filters = n_filters,n_residual_layers = n_residual_layers,
                 ratios = ratios, activation= activation, activation_params = activation_params, kernel_size=kernel_size,
                 last_kernel_size = last_kernel_size, residual_kernel_size = residual_kernel_size, dilation_base= dilation_base, causal=causal,
                 lstm = lstm,final_activation = final_activation,norm = norm,AffineTransform=FiLMDecoder,biLSTM = biLSTM,transformer = transformer,transformer_args={'max_seq_len':self.embeddingLength})
        self.inputSize = inputSize
        
        self.quantizationdim = quantizationdim
        self.n_q = n_q
        flattenShape = self.encoder(torch.randn(1,inputChannel,inputSize)).size(1), self.encoder(torch.randn(1,inputChannel,inputSize)).size(2)
        print('flattenShape:',flattenShape)
        quantizedchannel, quantizedlen = flattenShape[0] , flattenShape[1]
        flattenShape = flattenShape[0] * flattenShape[1]
        if self.quantizationdim is not None:
            if FixedCodeSize:
                if flattenShape % FixedCodeSize != 0:
                    factors = []
                    for i in range(1, flattenShape + 1):
                        if flattenShape % i == 0:
                            factors.append(i)
                    self.quantizationdim = int(flattenShape / min(factors, key=lambda x:abs(x-FixedCodeSize)))
                else:
                    self.quantizationdim = int(flattenShape / FixedCodeSize)
                print(f"Quantization Shape modified to {self.quantizationdim} with flattenShape of {flattenShape}")
            else:
                if flattenShape % quantizationdim != 0:
                    factors = []
                    for i in range(1, flattenShape + 1):
                        if flattenShape % i == 0:
                            factors.append(i)
                    self.quantizationdim = min(factors, key=lambda x:abs(x-quantizationdim))
                    print(f"Quantization Shape modified to {self.quantizationdim} with flattenShape of {flattenShape}")
            if bucketTransform:
                self.quantizationdim = int(quantizedchannel * NumBucket)
                if quantizedlen % NumBucket != 0:
                    self.pad = nn.ConstantPad1d((0,NumBucket - (quantizedlen % NumBucket)),0)
            self.bucketTransform = bucketTransform
            self.NumBuckets = NumBucket
            self.quantization = qt.ResidualVectorQuantization(num_quantizers=n_q, codebook_size=codebook_size,dim = self.quantizationdim, kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,threshold_ema_dead_code=threshold_ema_dead_code,l2norm = quantl2)
        else:
            self.quantization = nn.Identity()
        
        
        self.flattenQuantization = flattenQuantization
        if applyquantizerdroput:
            self.rng = np.random.default_rng(42)
        if encoders:
            self.mlpencoder = nn.Sequential(
                nn.utils.parametrizations.weight_norm(nn.Linear(int(flattenShape/self.quantizationdim), 16), name = 'weight'),
                nn.ELU(),
            )
            self.mlpdecoder = nn.Sequential(
                nn.utils.parametrizations.weight_norm(nn.Linear(16, int(flattenShape/self.quantizationdim)),name= 'weight'),
                nn.ELU(),
            )

    def forward(self,input,FiLM_input = None,returnAll = False,use_vq = True):
        x = self.encoder(input,FiLM_input)
        #print(x.shape)
        #if (FiLM_input is not None) & (hasattr(self,'FiLMlayers')):
            # if (self.FiLMparams['Last_layer_only']):
            #     FiLM_output1 = self.FiLMlayers[0](FiLM_input)
            #     FiLM_gamma1, FiLM_beta1 = FiLM_output1[:,0].unsqueeze(-1).unsqueeze(-1), FiLM_output1[:,1].unsqueeze(-1).unsqueeze(-1)
            #     FiLM_gamma1,FiLM_beta1 = FiLM_gamma1.repeat(1,x.shape[1],1), FiLM_beta1.repeat(1,x.shape[1],1)
            #     # print('gamma:',FiLM_gamma1.shape)
            #     # print('beta:',FiLM_beta1.shape)
            #     # print('embed:',x.shape)
            #     x = x * FiLM_gamma1 + FiLM_beta1
        if use_vq:
            # random dropout: https://kyutai.org/Moshi.pdf
            if self.quantizationdim is not None:
                if self.flattenQuantization:
                    original_shape = x.shape
                    x = x.view(x.shape[0],-1,1)
                elif self.bucketTransform:
                    if hasattr(self,'pad'):
                        self.unpadded_shape = x.shape
                        x = self.pad(x)
                    original_shape = x.shape
                    # u = torch.zeros(x.shape[0],self.quantizationdim,x.shape[2]//self.NumBuckets)
                    # for i in range(x.shape[2]):
                    #     dim1 = (i*original_shape[1])%self.quantizationdim
                    #     dim2 = dim1 + original_shape[1]
                    #     u[:,dim1:dim2,
                    #     (i*original_shape[1])//self.quantizationdim] += x[:,:,i]
                    # print(torch.eq(u,x.transpose(1,2).reshape(x.shape[0],x.shape[2]//self.NumBuckets,self.quantizationdim).transpose(1,2)).all())
                    # Underlying logic
                    #Input shape [B,C,L]
                    #Output shape [B,C*NumBuckets,L//NumBuckets]
                    x = x.transpose(1,2).reshape(x.shape[0],x.shape[2]//self.NumBuckets,self.quantizationdim).transpose(1,2)
                else:
                    original_shape = x.shape
                    x = x.view(x.shape[0],self.quantizationdim,-1)
                
                if hasattr(self,'mlpencoder'):
                    x = self.mlpencoder(x)
                BeforeQuantization = x

                if hasattr(self,'rng'):
                    if self.rng.random() >= 0.5: #quantizer dropout probability
                        num_quantizers = self.rng.integers(1,self.n_q+1)
                        quantized, out_Indices, out_losses = self.quantization(x,num_quantizers)
                    else:
                        quantized, out_Indices, out_losses = self.quantization(x)
                else:
                    quantized, out_Indices, out_losses = self.quantization(x)
                if hasattr(self,'mlpdecoder'):
                    quantized = self.mlpdecoder(quantized)
                if self.flattenQuantization:
                    quantized = quantized.view(original_shape[0],original_shape[1],original_shape[2])
                elif self.bucketTransform:
                    # u = torch.zeros(original_shape[0],original_shape[1],original_shape[2])
                    # for i in range(original_shape[2]):
                    #     dim1 = (i*original_shape[1])%self.quantizationdim
                    #     dim2 = dim1 + original_shape[1]
                    #     u[:,:,i] += quantized[:,dim1:dim2,
                    #                           (i*original_shape[1])//self.quantizationdim]
                    # print(torch.eq(u,quantized.transpose(1,2).reshape(original_shape[0],original_shape[2],original_shape[1]).transpose(1,2)).all())
                    quantized = quantized.transpose(1,2).reshape(original_shape[0],original_shape[2],original_shape[1]).transpose(1,2)
                    if hasattr(self,'unpadded_shape'):
                        quantized = quantized[:,:,:self.unpadded_shape[2]]
            else:
                quantized = x
                BeforeQuantization = None
                out_losses = 0
        else:
            quantized = x
            BeforeQuantization = None
            out_losses = 0
        x = self.decoder(quantized,FiLM_input)
        #print(x.shape)
        if x.shape[2] != input.shape[2]:
            #print(x.shape)
            x = x[:,:,:input.shape[2]]
        if returnAll:
            return x,out_losses, out_Indices
        return x,out_losses, BeforeQuantization
    
    def __getCodebook__(self,x):
        CodeBookList = []
        for layer in self.quantization.layers:
            CodeBookList.append(layer._codebook.embed)
        return torch.vstack(CodeBookList)
    
    def encode(self,x,numquantizer:int = 16, returnCodebookIndices:bool = False,returnAll = False):
        x = self.encoder(x,None)
        if self.quantizationdim is not None:
            if self.flattenQuantization:
                original_shape = x.shape
                x = x.view(x.shape[0],-1,1)
            elif self.bucketTransform:
                if hasattr(self,'pad'):
                    self.unpadded_shape = x.shape
                    x = self.pad(x)
                original_shape = x.shape
                x = x.transpose(1,2).reshape(x.shape[0],x.shape[2]//self.NumBuckets,self.quantizationdim).transpose(1,2)
            else:
                original_shape = x.shape
                x = x.view(x.shape[0],self.quantizationdim,-1)
            if hasattr(self,'mlpencoder'):
                x = self.mlpencoder(x)
            
        if returnAll:
            embedding, Indices, qLoss = self.quantization.encodeWithLoss(x,numquantizer)
            return embedding, Indices, qLoss, x

        if returnCodebookIndices:
            Indices = self.quantization.encode(x)
            return Indices
        else:
            return self.quantization(x)
        
    def decodeIndices(self,x):
        return self.quantization.decode(x)
    
    def decodeLatent(self,x):
        x = self.decoder(x)
        if self.inputSize != x.shape[2]:
            x = x[:,:,:self.inputSize]
        return x

    def reconstructIndices(self,x):
        x = self.decoder(self.quantization.decode(x))
        if self.inputSize != x.shape[2]:
            x = x[:,:,:self.inputSize]
        return x
    
    def forwardWithnumQuantizers(self,x,n_q:int):
        input_shape = x.shape
        x = self.encoder(x,None)
        if self.quantizationdim is not None:
            if self.flattenQuantization:
                original_shape = x.shape
                x = x.view(x.shape[0],-1,1)
            elif self.bucketTransform:
                if hasattr(self,'pad'):
                    self.unpadded_shape = x.shape
                    x = self.pad(x)
                original_shape = x.shape
                x = x.transpose(1,2).reshape(x.shape[0],x.shape[2]//self.NumBuckets,self.quantizationdim).transpose(1,2)
            else:
                original_shape = x.shape
                x = x.view(x.shape[0],self.quantizationdim,-1)
            if hasattr(self,'mlpencoder'):
                x = self.mlpencoder(x)
            BeforeQuantization = x
        embedding, _, qLoss = self.quantization(x,n_q)
        decoded = self.decoder(embedding)
        if decoded.shape[2] != input_shape[2]:
            decoded = decoded[:,:,:input_shape[2]]
        return decoded, qLoss, BeforeQuantization


if __name__ == '__main__':
    model_kwargs = {'inputChannel': 1, 'n_q': 22, 'n_filters': 32,'n_residual_layers': 1, 'lstm': 3, 'ratios':[10,7,6,5] , 'final_activation': 'Sigmoid',
                'residual_kernel_size': 8,'codebook_size':2048, 'FiLMEncoder' : False, 'FiLMDecoder' :False,'flattenQuantization' : False,
                'quantizationdim': 228, 'biLSTM': True,'causal':False,'useQINCo':False,'inputSize' : 13500,'dimension':228,'kernel_size':8, 
                'encoders':False,'bucketTransform':False}
    model_kwargs = {'inputChannel': 1, 'n_q':6, 'n_filters': 32,'n_residual_layers': 1, 'lstm': 2, 'ratios':[8,6,5,4] , 'final_activation': 'Sigmoid',
                'residual_kernel_size': 8,'codebook_size':1024, 'FiLMEncoder' : False, 'FiLMDecoder' :False,'flattenQuantization' : False,
                'quantizationdim': 256, 'biLSTM': True,'causal':False,'inputSize' : 13500,'dimension':256,'kernel_size':8, 
                'encoders':False,'bucketTransform':False,'applyquantizerdroput':True,'transformer':3}#480,480
    model = VQMSStream(**model_kwargs)
    model.to('cuda')
    print(model(torch.randn(16,1,13500,device='cuda'))[0].shape)
    # print(model.encode(torch.randn(16,1,13500),returnAll = True)[1].shape)
    # %%