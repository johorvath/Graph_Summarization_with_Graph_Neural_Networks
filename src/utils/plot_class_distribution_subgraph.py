"""
plot_class_distribution_subgraph.py: plot the class distribution of the given folds
by collecting the information from the single subgraphs
"""

import argparse
import os.path as osp
import configparser
import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

from neural_model import utils

from config_utils import config_util as cfg_u


import ast

from utils.plot_class_distribution import plot_class_distribution

import pickle 

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj (name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
        
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()
        
    cfg = configparser.ConfigParser()
    print(args)
    cfg.read(args.config)
    
    base_dir = cfg_u.makePath(cfg['DataExchange']['base_dir'])
    run_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['run_dir'])
    save_file = str(base_dir / cfg_u.makePath(cfg['DataExchange']['save_file']))
    
    plot_file_name =  cfg_u.makePath(cfg['WorkParameter']['plot_file_name'])
    plot_file_name_png = str(plot_file_name) + ".png"
    print(plot_file_name_png)
    plot_file_name_pgf = str(plot_file_name) + ".pgf"
    print(plot_file_name_pgf)
    
    folds =  ast.literal_eval(cfg.get('WorkParameter', 'folds'))
    desc_file = cfg.get( 'WorkParameter','desc_file' )
    
    data_folds = utils.load_data_fold( run_dir, folds, desc_file )
    
    print(data_folds)
    di = {}
        
    if osp.isfile(save_file + ".pkl"):
        di = load_obj( save_file )
        print("Loaded",len(di), "classes!")
    else:
        count = len(list(data_folds.iterrows()))    
        print("Total number:", count)
        classes = data_folds['class'].tolist()
        unique_class = list( set ( classes  ) )
        
        di = {x:  classes.count( x ) for x in unique_class }
        
        print(di)
                        
        save_obj(di, save_file)
        print("Number of classes: ", len(di))
    
    max_classes = cfg.getint('WorkParameter', 'max_classes')
        
    plot_class_distribution( di, filename = str( run_dir / plot_file_name_png), max_classes = max_classes )
    plot_class_distribution( di, filename = str( run_dir / plot_file_name_pgf), max_classes = max_classes )
 

if __name__ == "__main__":
    main()