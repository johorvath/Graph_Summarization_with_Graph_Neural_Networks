"""
eval_bloomfilter_reduced_schemex.py: Evaluate the Bloom filter results for the reduced Schemex dataset
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

import random 

from utils.bloom_metrics import rand_ind, imp_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    torch.set_num_threads(cfg.getint( 'WorkParameter','num_threads' ))
    
    min_support_classes = cfg.getint('WorkParameter', 'min_support_classes')
    mini_batch = cfg.getint('WorkParameter', 'mini_batch')
            
    base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'])
    
    graph_data =  base_dir / cfg_u.makePath(cfg['GNN']['save_file'])    
    k_fold = cfg.getint('GNN', 'k_fold')

    gs = gsg.graph_summary_generator( gs = cfg.getint('GraphSummary', 'GS-Model') )    
    gs.load( graph_data )       
    
    eval_value = ast.literal_eval(cfg.get('GraphSummary', 'eval'))
    
    save_fold_percentage = cfg.getfloat('GraphSummary', 'save_fold_percentage')
    
    print("Following metrics are calculated:", eval_value)
        
    print("Model: [","num_features:", gs.get_num_features(), "num_classes:",gs.get_num_classes(),"]")
    print("Number elements: ", len(gs.graph_information_))
    
    cl = {}
    
    for fold in range(k_fold):    
        random.seed(42)
        # sample x entries from k_folds[fold]
        new_size = int( len(gs.k_folds[fold]) * save_fold_percentage ) 
        k_folds_trunc = random.sample(gs.k_folds[fold], new_size )
        print("Reduce", len(gs.k_folds[fold]),"to", new_size)
        
        old = len(k_folds_trunc)        
        k_folds_trunc = [e for e in k_folds_trunc if len(gs.label_dict_[gs.graph_information_[e].hash]) > min_support_classes]        
        print("Reduce", old,"to", len(k_folds_trunc), "because of min_support ", min_support_classes)
        
        #Filter with mini-batch size
        degree_filtered_e = []
        for index, e in enumerate(k_folds_trunc):
            element = [e, gs.graph_information_[e]]
            degree = element[1].degree
            for index_j, j in enumerate(element[1].edges):                
                    if gs.graph_information_.get(j[1]) != None:
                        ob_el = (j[1], gs.graph_information_.get(j[1]))
                        degree += ob_el[1].degree 
            if degree < mini_batch -1:
                degree_filtered_e.append( e )
        print("Reduce", len(k_folds_trunc),"to", len(degree_filtered_e), "because of batch size ", mini_batch) 
        k_folds_trunc = degree_filtered_e
    
        for index, e in enumerate(k_folds_trunc):
            bloom = gs.graph_information_[e].bloomfilter
            
            if bloom in cl:
                cl[bloom].append(gs.graph_information_[e].hash)
            else:
                cl[bloom] = [gs.graph_information_[e].hash]
    
    if( "rand_index" in eval_value ):
        rand_index = rand_ind( gs ) 
        print("Overall rand_index: ", rand_index )
        
    if( "impurity" in eval_value or "acc" in eval_value):
        imp,acc = imp_acc( gs, cl ) 
        print("Total impurity: ", np.sum(np.array(imp)))
        print("Overall accuracy: ", acc )
        
   
        
if __name__ == "__main__":
    main()

