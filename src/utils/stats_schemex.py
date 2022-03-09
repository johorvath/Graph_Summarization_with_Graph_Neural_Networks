"""
stats_schemex.py: print stats for our custom reduced Schemex dataset of DylDO
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    torch.set_num_threads(cfg.getint( 'WorkParameter','num_threads' ))    
   
        
    base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'])

    
    graph_data =  base_dir / cfg_u.makePath(cfg['GraphSummary']['save_file'])    
    
    min_support_classes =  cfg.getint( 'WorkParameter','min_support_classes' )
    mini_batch =  cfg.getint( 'WorkParameter','mini_batch_size' )

    
    gs = gsg.graph_summary_generator( gs = cfg.getint('GraphSummary', 'GS-Model') )     
    
    
    gs.load( graph_data )
    
    k_folds_trunc = gs.graph_information_.keys()

    
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
            
        if index % 10 == 0:
            print("Filter mini-batch:",index,"of",len(k_folds_trunc))
    print("Reduce", len(k_folds_trunc),"to", len(degree_filtered_e), "because of batch size ", mini_batch) 
    k_folds_trunc = degree_filtered_e    

    
    subs = []
    preds = []
    objs = []
    spo = set([])
     
    for index, e in enumerate(k_folds_trunc):
        element = [e, gs.graph_information_[e]]
        for index_j, j in enumerate(element[1].edges):                
            spo.add((element[0],j[0],j[1]))
            if gs.graph_information_.get(j[1]) != None:
                ob_el = (j[1], gs.graph_information_.get(j[1]))
                for index_x, x in enumerate(ob_el[1].edges):                
                    spo.add((ob_el[0],x[0],x[1]))
                
        if index % 10 == 0:
            print("Filter mini-batch:",index,"of",len(k_folds_trunc))

    for s,p,o in spo:
        subs.append(s)
        preds.append(p)
        objs.append(o)
        
    print(preds)
      
    print("num vertices:",len(list(set(subs))))   
    print("num edges:", len(preds) ) 
    print("unique edges:",len(list(set(preds))))
    print("rdf_type:",preds.count('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'))
    

    
if __name__ == "__main__":
    main()

