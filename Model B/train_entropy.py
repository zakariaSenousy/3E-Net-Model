from src import *

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

import gc

if __name__ == '__main__':
    args = ModelOptions().parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
 
    pw_network_P2_1 = networks_P2_1.PatchWiseNetwork(args.channels)
    iw_network_P2_1 = networks_P2_1.ImageWiseNetwork(args.channels)
    
    pw_network_P2_2 = networks_P2_2.PatchWiseNetwork(args.channels)
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
    #gc.collect()
    
    pw_network_P6_2 = networks_P6_2.PatchWiseNetwork(args.channels)
    iw_network_P6_2 = networks_P6_2.ImageWiseNetwork(args.channels)
    #gc.collect()
    
    pw_network_P7 = networks_P7.PatchWiseNetwork(args.channels)
    iw_network_P7 = networks_P7.ImageWiseNetwork(args.channels)
    gc.collect()
        
    pw_network_P8 = networks_P8.PatchWiseNetwork(args.channels)
    iw_network_P8 = networks_P8.ImageWiseNetwork(args.channels)
    gc.collect()
        
    pw_network_P9 = networks_P9.PatchWiseNetwork(args.channels)
    iw_network_P9 = networks_P9.ImageWiseNetwork(args.channels)
    gc.collect()
        
    pw_network_P10 = networks_P10.PatchWiseNetwork(args.channels)
    iw_network_P10 = networks_P10.ImageWiseNetwork(args.channels)
    gc.collect()
        
    pw_network_P11 = networks_P11.PatchWiseNetwork(args.channels)
    iw_network_P11 = networks_P11.ImageWiseNetwork(args.channels)
    gc.collect()
        
    pw_network_P12 = networks_P12.PatchWiseNetwork(args.channels)
    iw_network_P12 = networks_P12.ImageWiseNetwork(args.channels)
     
  
    
    if args.network == '0' or args.network == '1':
        pw_model = models_P2_1.PatchWiseModel(args, pw_network_P2_1)
        pw_model.train()
        
    
    if args.network == '0' or args.network == '2':
              
        iw_model_P2_1 = models_P2_1.ImageWiseModel(args, iw_network_P2_1, pw_network_P2_1)
        iw_model_P2_1.train()
    
        iw_model_P2_2 = models_P2_2.ImageWiseModel(args, iw_network_P2_2, pw_network_P2_2)
        iw_model_P2_2.train()
        
        iw_model_P2_3 = models_P2_3.ImageWiseModel(args, iw_network_P2_3, pw_network_P2_3)
        iw_model_P2_3.train()

        iw_model_P3_1 = models_P3_1.ImageWiseModel(args, iw_network_P3_1, pw_network_P3_1)
        iw_model_P3_1.train()
        
        iw_model_P3_2 = models_P3_2.ImageWiseModel(args, iw_network_P3_2, pw_network_P3_2)
        iw_model_P3_2.train()
            
        iw_model_P4_1 = models_P4_1.ImageWiseModel(args, iw_network_P4_1, pw_network_P4_1)
        iw_model_P4_1.train()
        
        iw_model_P4_2 = models_P4_2.ImageWiseModel(args, iw_network_P4_2, pw_network_P4_2)
        iw_model_P4_2.train()
        
        iw_model_P5_1 = models_P5_1.ImageWiseModel(args, iw_network_P5_1, pw_network_P5_1)
        iw_model_P5_1.train()
        
        iw_model_P5_2 = models_P5_2.ImageWiseModel(args, iw_network_P5_2, pw_network_P5_2)
        iw_model_P5_2.train()
        
        iw_model_P6_1 = models_P6_1.ImageWiseModel(args, iw_network_P6_1, pw_network_P6_1)
        iw_model_P6_1.train()
        #gc.collect()
        
        iw_model_P6_2 = models_P6_2.ImageWiseModel(args, iw_network_P6_2, pw_network_P6_2)
        iw_model_P6_2.train()
        #gc.collect()
            
        iw_model_P7 = models_P7.ImageWiseModel(args, iw_network_P7, pw_network_P7)
        iw_model_P7.train()       
        gc.collect()
        
        iw_model_P8 = models_P8.ImageWiseModel(args, iw_network_P8, pw_network_P8)
        iw_model_P8.train()        
        gc.collect()
        
        iw_model_P9 = models_P9.ImageWiseModel(args, iw_network_P9, pw_network_P9)
        iw_model_P9.train()
        gc.collect()
        
        iw_model_P10 = models_P10.ImageWiseModel(args, iw_network_P10, pw_network_P10)
        iw_model_P10.train()
        gc.collect()
        
        iw_model_P11 = models_P11.ImageWiseModel(args, iw_network_P11, pw_network_P11)
        iw_model_P11.train()
        gc.collect()
        
        iw_model_P12 = models_P12.ImageWiseModel(args, iw_network_P12, pw_network_P12)
        iw_model_P12.train()
        gc.collect()
        