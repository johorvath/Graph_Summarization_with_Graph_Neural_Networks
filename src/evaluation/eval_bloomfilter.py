"""
eval_bloomfilter.py: Evaluate the bloom filter results
for more information see Graph Summarization with Graph Neural Networks - Scientific Paper
"""
import argparse
import configparser
import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

import torch

from config_utils import config_util as cfg_u

from graph_summary_generator import graph_summary_generator as gsg

import ast

import numpy as np

from utils.bloom_metrics import rand_ind, imp_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    torch.set_num_threads(cfg.getint( 'WorkParameter','num_threads' ))
            
    base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'])
    
    graph_data =  base_dir / cfg_u.makePath(cfg['GNN']['save_file'])    

    gs = gsg.graph_summary_generator( gs = cfg.getint('GraphSummary', 'GS-Model') )    
    gs.load( graph_data )       
    
    eval_value = ast.literal_eval(cfg.get('GraphSummary', 'eval'))
    
    print("Following metrics are calculated:", eval_value)
        
    print("Model: [","num_features:", gs.get_num_features(), "num_classes:",gs.get_num_classes(),"]")
    print("Number elements: ", len(gs.graph_information_))
    
    cl = {}
    for i in gs.graph_information_:
        bloom = gs.graph_information_[i].bloomfilter
        
        if bloom in cl:
            cl[bloom].append(gs.graph_information_[i].hash)
        else:
            cl[bloom] = [gs.graph_information_[i].hash]
    
    if( "rand_index" in eval_value ):
        rand_index = rand_ind( gs ) 
        print("Overall rand_index: ", rand_index )
        
    if( "impurity" in eval_value or "acc" in eval_value):
        imp,acc = imp_acc( gs, cl ) 
        print("Total impurity: ", np.sum(np.array(imp)))
        print("Overall accuracy: ", acc )
        
   
        
if __name__ == "__main__":
    main()

