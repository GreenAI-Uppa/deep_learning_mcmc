import collections
from abc import ABC, abstractmethod
import torch
import numpy as np
from deep_learning_mcmc import nets
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import copy
import tflite_runtime.interpreter as tflite

def get_interpreter(tflite_model):
    # Load TFLite model and allocate tensors.
    #interpreter = tf.lite.Interpreter(model_content=tflite_model)
    #print(tflite_model)
    
    interpreter = tflite.Interpreter(model_content=tflite_model)
    #interpreter = tf.lite.Interpreter(model_path="BinaryMcmc.tflite")
    # Note: need to fake resize the input & reallocate tensors
    interpreter.resize_tensor_input(0, [1,3,32,32], strict=True)
    interpreter.allocate_tensors()
    return interpreter

def update(tflite_model,model,neighborhood, proposal,update_b = True):
    interpreter = get_interpreter(tflite_model)
    layer_idx, idces = neighborhood
    if update_b:
        w = model.layers[layer_idx].update(idces, proposal)
    else :
        w =  model.layers[layer_idx].undo(idces, proposal)
    if w != None:
        wt = w.cpu().detach().numpy()
        # get weight tensor to modify
    
        tensors_quant = None
        #print(interpreter.get_tensor_details())
        for details in interpreter.get_tensor_details():
            if 'sequential/quant_conv2d/' in details['name'] : 
                tensors_quant = details
                break
        wt = np.transpose(np.array(wt), (0,2,3,1))
        print("#########")
        print(wt.shape)
        print(interpreter.get_tensor(tensors_quant['index']).shape)
        interpreter.set_tensor(tensors_quant['index'], wt)
    return interpreter

def undo(tflite_model,model,neighborhood, proposal,update_b = False):
    return update(tflite_model,neighborhood, proposal,update_b)

def predict(interpreter,X):
     # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    pred =None
    for x in X :
        interpreter.set_tensor(input_details[0]['index'], np.array([np.array(x)]))
        interpreter.invoke()
        if pred == None:
            pred = torch.tensor(interpreter.get_tensor(output_details[0]['index']))
        else :

            pred = torch.cat((pred,torch.tensor(interpreter.get_tensor(output_details[0]['index']))),0)

    return pred

