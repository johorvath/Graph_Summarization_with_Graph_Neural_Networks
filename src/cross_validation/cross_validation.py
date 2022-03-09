"""
cross_validation.py: create a model (mlp,gcn,sgcn,rgcn,graphsage,graphsaint) and run a k-fold
crossvalidation. The results are stored in a csv-file.
for more information see GNN as Graph Summaries - Technical Report
"""


import argparse
import os
import configparser
import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

import torch
from neural_model import mlp, utils, rgcn, gcn, sage, sgcn, saint

from config_utils import config_util as cfg_u

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import pandas as pd

import numpy as np


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()
        
    cfg = configparser.ConfigParser()
    print(args)
    cfg.read(args.config)
    
    base_dir = cfg_u.makePath(cfg['DataExchange']['base_dir'])
    run_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['run_dir'])
    load_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['load_dir'])
    save_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['save_dir'])
    
    if(cfg.getint( 'WorkParameter','num_threads' ) > 0 ):
        torch.set_num_threads(cfg.getint( 'WorkParameter','num_threads' ))
        
    load_checkpoint = cfg.getboolean('WorkParameter', 'load_checkpoint')
    run_test = cfg.getboolean('WorkParameter', 'run_test')
    
    cuda_core = cfg.get('WorkParameter', 'cuda_core')    
        
    gs_model = str(run_dir.stem)
        
    val_sample_size = cfg.getint('WorkParameter', 'val_sample_size')
    val_step = cfg.getint('WorkParameter', 'val_step')
    
    test_sample_size = cfg.getint('WorkParameter', 'test_sample_size')
    
    k_fold= cfg.getint('WorkParameter', 'k_fold')
    
    dsc_file = cfg.get('WorkParameter', 'description_file')
    load_dsc_file = cfg.get('WorkParameter', 'load_description_file')
    
    epochs = cfg.getint('WorkParameter', 'epochs')
    guard = cfg.getint('WorkParameter', 'guard')
    
    model_name =  cfg.get('GNN', 'model_name')
    
    learning_rate =  cfg.getfloat('GNN', 'learning_rate')
    weight_decay =  cfg.getfloat('GNN', 'weight_decay')
    dropout =  cfg.getfloat('GNN', 'dropout')
    
    hidden_layer =  cfg.getint('GNN', 'hidden_layer')
    

    num_features =  cfg.getint('GNN', 'num_features')
    num_classes =  cfg.getint('GNN', 'num_classes')
        
    device = torch.device(cuda_core if torch.cuda.is_available() else 'cpu')
    print("Running on ", device)
        
    csv_file =  str(cfg_u.makePath("result_k_cross_" + gs_model + "_" + model_name ) ) + ".csv"
    csv_dir = save_dir / cfg_u.makePath("k_folds")

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        
    csv_file = csv_dir / csv_file
    result = {"test_acc":[]}
    
    print("Load data!")
    load_data = utils.load_data_fold( load_dir, list(range(k_fold)), load_dsc_file )

    for i in range(k_fold):
        fold = list(range(k_fold))
        val_index = i
        test_index = (i+1)%k_fold
        
        val_fold = [fold[val_index]]
        test_fold = [fold[test_index]]
        fold = list(set(fold) - set(val_fold))
        fold = list(set(fold) - set(test_fold))
        
        train_fold = fold
    
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        s =  current_time + "_" + str(cfg_u.makePath(cfg['WorkParameter']['summary_file']+ "_" 
                                                     + gs_model + "_" + model_name + "_" 
                                                     + str(learning_rate).replace(".", "-") + "_" + str(hidden_layer)
                                                     + "_" + str(dropout).replace(".", "-") + "_" + str(i)))
        
        s1 =  current_time + "_" + str(cfg_u.makePath(cfg['WorkParameter']['checkpoint_file']+ "_" 
                                                     + gs_model + "_" + model_name + "_" 
                                                     + str(learning_rate).replace(".", "-") + "_" + str(hidden_layer)
                                                     + "_" + str(dropout).replace(".", "-") + "_" + str(i)))
        
        checkpoint_file =  str(save_dir / s1 )
        
        writer_dir = save_dir / cfg_u.makePath("runs") / s
        writer = SummaryWriter( str(writer_dir) )
                
        
        print("Start loading train data!")
        train_data = utils.load_fold( run_dir, train_fold, load_data, dsc_file )
        print("Start loading val data!")
        val_data = utils.load_fold( run_dir, val_fold, load_data, dsc_file )
        print("Start loading test data!")
        test_data = utils.load_fold( run_dir, test_fold, load_data, dsc_file )
            
        if ( model_name == "mlp" ):        
            model = mlp.MLP( num_features, num_classes, hidden_layer, dropout = dropout )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif ( model_name == "gcn" ):
            model = gcn.GCN( num_features, num_classes, hidden_layer, dropout = dropout )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif model_name == "rgcn":
            model = rgcn.RGCN( guard,num_classes,num_features
                              , hidden_channels = hidden_layer)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif model_name == "sgcn":
            if hidden_layer == 0:
                model = sgcn.SGCN( num_features, num_classes, K=1 )
            else:
                model = sgcn.SGCN( num_features, num_classes, K=2 )
            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)        
        elif model_name == "graphsaint":
            model = saint.SAINT( num_features, num_classes, hidden_layer, dropout = dropout )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)        
        elif model_name == "graphsage":
            model = sage.SAGE( num_features, num_classes, hidden_layer, dropout = dropout )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
        print("Model:",model)
        print("Parameter count:", sum(p.numel() for p in model.parameters()))        
        print("Optimizer:",optimizer)
        model = model.to(device)
                
        
        if( load_checkpoint ):
            #https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])  
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            data_list = {"train":train_data,"val":val_data}
            utils.stichted_training( device, data_list, model, optimizer, writer, checkpoint_file, epochs, guard, val_sample_size, val_step )
            
        if run_test:
            result["test_acc"].append( utils.test_model( model, test_data, device, writer, test_sample_size, guard ) )
            
        df = pd.DataFrame(data=result)
        df.to_csv(csv_file, index=False, encoding='utf8')
        print("--------")
        del model
        del optimizer
        torch.cuda.empty_cache() 
        
    acc = np.array( df["test_acc"] )
    
    print(csv_file, "[mean:", np.mean(acc, axis=0), "std_error:", np.std(acc, axis=0)/np.sqrt(len(acc)),"]")
    
    
    
if __name__ == "__main__":
    main()

