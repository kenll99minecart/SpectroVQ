from . import modelStream
import torch
import yaml
DEFAULTPARAMS = {'inputChannel': 1, 'n_q':12, 'n_filters': 32,'n_residual_layers': 1, 'lstm': 0, 'ratios':[9,7,4,2] , 'final_activation': 'Sigmoid',
                'residual_kernel_size': 6,'codebook_size':1024, 'FiLMEncoder' : False, 'FiLMDecoder' :False,'flattenQuantization' : False,
                'quantizationdim': 256, 'biLSTM': False,'causal':False,'useQINCo':False,'inputSize' : 13500,'dimension':256,'kernel_size':8, 
                'encoders':False,'bucketTransform':False,'applyquantizerdroput':False,'transformer':6}
def getModel(yaml_file=None):
    '''
    Load the model from the yaml file

    Parameters:
    yaml_file (str): The path to the yaml file; if None, the default parameters are used
    weights_file (str): The path to the weights file; if None, the weights will not be loaded
    '''
    if yaml_file is None:
        model_kwargs = DEFAULTPARAMS
        
    else:
        with open(yaml_file) as f:
            model_kwargs = yaml.load(f, Loader=yaml.FullLoader)

    SpectraStream = modelStream.VQMSStream(**model_kwargs) # Following the original parameters
    return SpectraStream

def ApplyWeights(model,weights_file):
    '''
    Load the model from the yaml file
    
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
