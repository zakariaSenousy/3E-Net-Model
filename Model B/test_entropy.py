from src import *
import numpy as np

from src import models_P2_1,networks_P2_1
from src import models_P2_2,networks_P2_2
from src import models_P2_3,networks_P2_3
from src import models_P3_1,networks_P3_1
from src import models_P3_2,networks_P3_2
from src import models_P4_1,networks_P4_1
from src import models_P4_2,networks_P4_2
from src import models_P5_1,networks_P5_1
from src import models_P5_2,networks_P5_2
from src import models_P6_1,networks_P6_1
from src import models_P6_2,networks_P6_2
from src import models_P7,networks_P7
from src import models_P8,networks_P8
from src import models_P9,networks_P9
from src import models_P10,networks_P10
from src import models_P11,networks_P11
from src import models_P12,networks_P12


#------------------------

import torch
from torch.distributions import Categorical


args = ModelOptions().parse()


LABELS = ['Grade_1', 'Grade_2', 'Grade_3']
LabelAbbrev = ['G1', 'G2', 'G3']


def Single_Feature_extractor(p, mode='test'):
    results = []
    for i in range(len(p)):     
        print(p[i][0])
        model_pred = np.array(p[i][0])
        highest_index = 2 - np.argmax(model_pred[::-1])
        label = LABELS[highest_index]
        image_name = p[i][1]
        results.append([label,image_name])
    
    print('Feature Extractor model:')
    print('--------------------------------------')
    for i in results:
        print(i)
    
    if mode == 'valid':
        correct = 0
        for i in range(len(results)):
            if results[i][0] == LABELS[0] and LabelAbbrev[0] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[1] and LabelAbbrev[1] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[2] and LabelAbbrev[2] in results[i][1]:
                correct+=1
        print('Correct: ', correct)
        print('Total: ', len(results))
        val_acc = (correct / len(results)) *100
        print('Validation Accuracy = ', val_acc,'%')
        
    
        
def Single_ContextAware(p, mode = 'test'):
    results = []
    for i in range(len(p)):     
        model_pred = np.array(p[i][0])
        std = np.array(p[i][1])
        #logp = np.log2(model_pred)
        #entropy = np.sum(-model_pred * logp)
        highest_index = np.argmax(model_pred)
        label = LABELS[highest_index]
        image_name = p[i][3]
        results.append([label, image_name, model_pred, std])
    
    print('Context Aware model:')
    print('--------------------------------------')
    for i in results:
        print(i)
    
    if mode == 'valid':
        correct = 0
        for i in range(len(results)):
            if results[i][0] == LABELS[0] and LabelAbbrev[0] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[1] and LabelAbbrev[1] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[2] and LabelAbbrev[2] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[3] and LabelAbbrev[3] in results[i][1]:
                correct+=1
        val_acc = (correct / len(results)) *100
        print('Validation Accuracy = ', val_acc,'%')
    
        
def Feature_extractor_Ensemble (p, p1, p2, mode = 'test'):
    ens_results = []
    for i in range(len(p)):     
        model_pred = np.array(p[i][0])
        model_pred1 = np.array(p1[i][0])
        model_pred2 = np.array(p2[i][0])
        #model_pred3 = np.array(p3[i][0])
        #model_pred4 = np.array(p4[i][0])
        final = np.divide(model_pred + model_pred1 + model_pred2, 3)
        highest_index = 3 - np.argmax(final[::-1])
        label = LABELS[highest_index]
        image_name = p[i][1]
        ens_results.append([label,image_name])
    
    print('Ensemble of Feature Extractor models:')
    print('--------------------------------------')
    for i in ens_results:
        print(i)
    
    if mode == 'valid':
        correct = 0
        for i in range(len(ens_results)):
            if ens_results[i][0] == LABELS[0] and LabelAbbrev[0] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[1] and LabelAbbrev[1] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[2] and LabelAbbrev[2] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[3] and LabelAbbrev[3] in ens_results[i][1]:
                correct+=1
        val_acc = (correct / len(ens_results)) *100
        print('Validation Accuracy = ', val_acc,'%')
    
     



def ContextAware_Ensemble (p1, p2, p3, p4, p5, p6, p7, p8,
                           p9, p10, p11, p12, p13, p14, p15, p16, p17,
                           threshold = 0, mode = 'test'):
    ens_results = []
    excluded_imgs = []   
    for i in range(len(p1)):
        chosen_models = []  
        #1
        p_tensor = torch.Tensor(np.array(p1[i][0]))
        entropy = Categorical(probs = p_tensor).entropy() 
        if entropy < threshold:
            chosen_models.append(np.array(p1[i][0]))
            
        #2
        p_tensor = torch.Tensor(np.array(p2[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()
        if entropy < threshold:
            chosen_models.append(np.array(p2[i][0]))
        
        #3
        p_tensor = torch.Tensor(np.array(p3[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()
        if entropy < threshold:
            chosen_models.append(np.array(p3[i][0]))
        
        #4
        p_tensor = torch.Tensor(np.array(p4[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()     
        if entropy < threshold:
            chosen_models.append(np.array(p4[i][0]))
            
        #5
        p_tensor = torch.Tensor(np.array(p5[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()   
        if entropy < threshold:
            chosen_models.append(np.array(p5[i][0]))
            
        #6
        p_tensor = torch.Tensor(np.array(p6[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()     
        if entropy < threshold:
            chosen_models.append(np.array(p6[i][0]))
            
        #7
        p_tensor = torch.Tensor(np.array(p7[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()  
        if entropy < threshold:
            chosen_models.append(np.array(p7[i][0]))
            
        #8
        p_tensor = torch.Tensor(np.array(p8[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()  
        if entropy < threshold:
            chosen_models.append(np.array(p8[i][0]))
            
        #9
        p_tensor = torch.Tensor(np.array(p9[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()   
        if entropy < threshold:
            chosen_models.append(np.array(p9[i][0]))
            
        #10
        p_tensor = torch.Tensor(np.array(p10[i][0]))
        entropy = Categorical(probs = p_tensor).entropy() 
        if entropy < threshold:
            chosen_models.append(np.array(p10[i][0]))
        
        #11
        p_tensor = torch.Tensor(np.array(p11[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()  
        if entropy < threshold:
            chosen_models.append(np.array(p11[i][0]))
        
        #12
        p_tensor = torch.Tensor(np.array(p12[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()  
        if entropy < threshold:
            chosen_models.append(np.array(p12[i][0]))
            
        #13
        p_tensor = torch.Tensor(np.array(p13[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()     
        if entropy < threshold:
            chosen_models.append(np.array(p13[i][0]))
            
        #14
        p_tensor = torch.Tensor(np.array(p14[i][0]))
        entropy = Categorical(probs = p_tensor).entropy() 
        if entropy < threshold:
            chosen_models.append(np.array(p14[i][0]))
        
        #15
        p_tensor = torch.Tensor(np.array(p15[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()  
        if entropy < threshold:
            chosen_models.append(np.array(p15[i][0]))
            
        #16
        p_tensor = torch.Tensor(np.array(p16[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()    
        if entropy < threshold:
            chosen_models.append(np.array(p16[i][0]))
            
        #17
        p_tensor = torch.Tensor(np.array(p17[i][0]))
        entropy = Categorical(probs = p_tensor).entropy()    
        if entropy < threshold:
            chosen_models.append(np.array(p17[i][0]))
            
            
        #----------------------------------------
        if len(chosen_models) != 0:
            final = np.sum(chosen_models, axis=0)        
            highest_index = np.argmax(final)
            label = LABELS[highest_index]
            image_name = p1[i][2]        
            ens_results.append([label, image_name, final, len(chosen_models)])
        
        if len(chosen_models) == 0:
            exc_image_name = p1[i][2]
            excluded_imgs.append([exc_image_name, len(chosen_models)])  
            
    print('Ensemble of Context-aware models:')
    print('--------------------------------------')
    for i in ens_results:
        print(i)
    
    if mode == 'valid':
        correct = 0
        for i in range(len(ens_results)):
            if ens_results[i][0] == LABELS[0] and LabelAbbrev[0] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[1] and LabelAbbrev[1] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[2] and LabelAbbrev[2] in ens_results[i][1]:
                correct+=1
            
        print ('Included Images: ', len(ens_results))
        exc = len(p1) - len(ens_results)
        print('Excluded Images: ', exc)
        if len(ens_results) != 0:
            val_acc = (correct / len(ens_results)) *100
            print('Validation Accuracy = ', val_acc,'%')
    
    print('EXCLUDED IMGS')
    print('-------------------------')
    for j in excluded_imgs:
        print(j)
    print('-------------------------')
    
    

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


pw_network_P2_1 = networks_P2_1.PatchWiseNetwork(args.channels)
iw_network_P2_1 = networks_P2_1.ImageWiseNetwork(args.channels)
    
"""pw_network_P2_2 = networks_P2_2.PatchWiseNetwork(args.channels)
iw_network_P2_2 = networks_P2_2.ImageWiseNetwork(args.channels)
    
pw_network_P2_3 = networks_P2_3.PatchWiseNetwork(args.channels)
iw_network_P2_3 = networks_P2_3.ImageWiseNetwork(args.channels)    
    
pw_network_P3_1 = networks_P3_1.PatchWiseNetwork(args.channels)
iw_network_P3_1 = networks_P3_1.ImageWiseNetwork(args.channels)
    
pw_network_P3_2 = networks_P3_2.PatchWiseNetwork(args.channels)
iw_network_P3_2 = networks_P3_2.ImageWiseNetwork(args.channels)
        
pw_network_P4_1 = networks_P4_1.PatchWiseNetwork(args.channels)
iw_network_P4_1 = networks_P4_1.ImageWiseNetwork(args.channels)
    
pw_network_P4_2 = networks_P4_2.PatchWiseNetwork(args.channels)
iw_network_P4_2 = networks_P4_2.ImageWiseNetwork(args.channels)
   
pw_network_P5_1 = networks_P5_1.PatchWiseNetwork(args.channels)
iw_network_P5_1 = networks_P5_1.ImageWiseNetwork(args.channels)
    
pw_network_P5_2 = networks_P5_2.PatchWiseNetwork(args.channels)
iw_network_P5_2 = networks_P5_2.ImageWiseNetwork(args.channels)
    
pw_network_P6_1 = networks_P6_1.PatchWiseNetwork(args.channels)
iw_network_P6_1 = networks_P6_1.ImageWiseNetwork(args.channels)
    
pw_network_P6_2 = networks_P6_2.PatchWiseNetwork(args.channels)
iw_network_P6_2 = networks_P6_2.ImageWiseNetwork(args.channels)
    
pw_network_P7 = networks_P7.PatchWiseNetwork(args.channels)
iw_network_P7 = networks_P7.ImageWiseNetwork(args.channels)
    
pw_network_P8 = networks_P8.PatchWiseNetwork(args.channels)
iw_network_P8 = networks_P8.ImageWiseNetwork(args.channels)
    
pw_network_P9 = networks_P9.PatchWiseNetwork(args.channels)
iw_network_P9 = networks_P9.ImageWiseNetwork(args.channels)
    
pw_network_P10 = networks_P10.PatchWiseNetwork(args.channels)
iw_network_P10 = networks_P10.ImageWiseNetwork(args.channels)
    
pw_network_P11 = networks_P11.PatchWiseNetwork(args.channels)
iw_network_P11 = networks_P11.ImageWiseNetwork(args.channels)
    
pw_network_P12 = networks_P12.PatchWiseNetwork(args.channels)
iw_network_P12 = networks_P12.ImageWiseNetwork(args.channels)"""

#--------------------------------------------------------------

if args.testset_path is '':
    import tkinter.filedialog as fdialog

    args.testset_path = fdialog.askopenfilename(initialdir=r"./dataset/test", title="choose your file", filetypes=(("tiff files", "*.tif"), ("all files", "*.*")))

if args.network == '1':
    pw_model_P2_1 = models_P2_1.PatchWiseModel(args, pw_network_P2_1)
    #pred = pw_model_P2_1.test(args.testset_path)
    vis = pw_model_P2_1.visualize_last(args.testset_path)
  
    #Feature_extractor_Ensemble(pred1, pred2, pred3, pred4, mode = 'test')
    #Single_Feature_extractor(pred, mode = 'valid')
    
else:
    
    im_model_P2_1 = models_P2_1.ImageWiseModel(args, iw_network_P2_1, pw_network_P2_1)
    context_P2_1 = im_model_P2_1.test(args.testset_path, ensemble= False)
    
    im_model_P2_2 = models_P2_2.ImageWiseModel(args, iw_network_P2_2, pw_network_P2_2)
    context_P2_2 = im_model_P2_2.test(args.testset_path, ensemble= False)
    
    im_model_P2_3 = models_P2_3.ImageWiseModel(args, iw_network_P2_3, pw_network_P2_3)
    context_P2_3 = im_model_P2_3.test(args.testset_path, ensemble= False)
    
    
    im_model_P3_1 = models_P3_1.ImageWiseModel(args, iw_network_P3_1, pw_network_P3_1)
    context_P3_1 = im_model_P3_1.test(args.testset_path, ensemble= False)
    
    im_model_P3_2 = models_P3_2.ImageWiseModel(args, iw_network_P3_2, pw_network_P3_2)
    context_P3_2 = im_model_P3_2.test(args.testset_path, ensemble= False)
    
    
    im_model_P4_1 = models_P4_1.ImageWiseModel(args, iw_network_P4_1, pw_network_P4_1)
    context_P4_1 = im_model_P4_1.test(args.testset_path, ensemble= False)
    
    im_model_P4_2 = models_P4_2.ImageWiseModel(args, iw_network_P4_2, pw_network_P4_2)
    context_P4_2 = im_model_P4_2.test(args.testset_path, ensemble= False)
    
    
    im_model_P5_1 = models_P5_1.ImageWiseModel(args, iw_network_P5_1, pw_network_P5_1)
    context_P5_1 = im_model_P5_1.test(args.testset_path, ensemble= False)
    
    im_model_P5_2 = models_P5_2.ImageWiseModel(args, iw_network_P5_2, pw_network_P5_2)
    context_P5_2 = im_model_P5_2.test(args.testset_path, ensemble= False)
    
    im_model_P6_1 = models_P6_1.ImageWiseModel(args, iw_network_P6_1, pw_network_P6_1)
    context_P6_1 = im_model_P6_1.test(args.testset_path, ensemble= False)
    
    im_model_P6_2 = models_P6_2.ImageWiseModel(args, iw_network_P6_2, pw_network_P6_2)
    context_P6_2 = im_model_P6_2.test(args.testset_path, ensemble= False)    
    
    im_model_P7 = models_P7.ImageWiseModel(args, iw_network_P7, pw_network_P7)
    context_P7 = im_model_P7.test(args.testset_path, ensemble= False)
    
    im_model_P8 = models_P8.ImageWiseModel(args, iw_network_P8, pw_network_P8)
    context_P8 = im_model_P8.test(args.testset_path, ensemble= False)
    
    im_model_P9 = models_P9.ImageWiseModel(args, iw_network_P9, pw_network_P9)
    context_P9 = im_model_P9.test(args.testset_path, ensemble= False)
    
    im_model_P10 = models_P10.ImageWiseModel(args, iw_network_P10, pw_network_P10)
    context_P10 = im_model_P10.test(args.testset_path, ensemble= False)
    
    im_model_P11 = models_P11.ImageWiseModel(args, iw_network_P11, pw_network_P11)
    context_P11 = im_model_P11.test(args.testset_path, ensemble= False)
    
    im_model_P12 = models_P12.ImageWiseModel(args, iw_network_P12, pw_network_P12)
    context_P12 = im_model_P12.test(args.testset_path, ensemble= False)
    
    
    
    #--------------------------
    
    
    t = np.array([0.00000001, 0.00000005, 0.00000009,
                  0.0000001, 0.0000005, 0.0000009,
                  0.000001, 0.000005, 0.000009, 
                  0.00001, 0.00005, 0.00009, 
                  0.0001, 0.0005, 0.0009, 
                  0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                  0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5])
    for val in t:
        print('THRESHOLD = ', val)
        ContextAware_Ensemble(context_P2_1, context_P2_2, context_P2_3,  
                              context_P3_1, context_P3_2, 
                              context_P4_1, context_P4_2,
                              context_P5_1, context_P5_2, context_P6_1, context_P6_2,
                              context_P7, context_P8, context_P9, context_P10, context_P11,
                              context_P12, threshold = val, mode = 'valid')

    
    
                          
    #Single_ContextAware(context_N1, mode = 'valid')