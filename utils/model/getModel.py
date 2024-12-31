from . import modelStream
import torch
import json
DEFAULTPARAMS = {'inputChannel': 1, 'n_q': 16, 'n_filters': 32,'n_residual_layers': 1, 'lstm': 2, 'ratios':[9,7,6,2] , 'final_activation': 'Sigmoid',
                'residual_kernel_size': 8,'codebook_size':1024, 'FiLMEncoder' : False, 'FiLMDecoder' :False,'flattenQuantization' : False,
                'quantizationdim': 512, 'biLSTM': True,'causal':False,'useQINCo':False,'inputSize' : 13500,'dimension':512,'kernel_size':8, 
                'encoders':False,'bucketTransform':False,'applyquantizerdroput':False}

def getModel(json_file=None):
    '''
    Load the model from the json file
    
    Parameters:
    json_file (str): The path to the json file; if None, the default parameters are used
    weights_file (str): The path to the weights file; if None, the weights will not be loaded
    '''
    if json_file is None:
        model_kwargs = DEFAULTPARAMS
        
    else:
        with open(json_file) as f:
            model_kwargs = json.load(f)
    
    SpectraStream = modelStream.VQMSStream(**model_kwargs) # Following the original parameters 
    return SpectraStream

def ApplyWeights(model,weights_file):
    '''
    Load the model from the json file
    
    Parameters:
    model (torch.nn.Module): The model to which the weights will be applied
    weights_file (str): The path to the weights file; if None, the weights will not be loaded
    '''
    weights = torch.load(weights_file)
    weights_dict = {}
    for k, v in weights['state_dict'].items():
        new_k = k.replace('module.', '') if 'module' in k else k
        new_k = '.'.join(new_k.split('.')[1:]) if 'model' in new_k else new_k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict)
    return model
