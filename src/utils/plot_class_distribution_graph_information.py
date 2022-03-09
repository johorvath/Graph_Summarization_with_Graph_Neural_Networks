"""
plot_class_distribution_graph_information.py: plot the class distribution of the given
graph subject information dict
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


import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    torch.set_num_threads(cfg.getint( 'WorkParameter','num_threads' ))
        
    base_dir = cfg_u.makePath(cfg['DataExchange']['base_dir'])
    run_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['run_dir'])
    
    graph_data =  base_dir / cfg_u.makePath(cfg['GraphSummary']['save_file'])  
    plot_file_name =  cfg_u.makePath(cfg['WorkParameter']['plot_file_name'])
    plot_file_name_png = str(plot_file_name) + ".png"
    print(plot_file_name_png)
    plot_file_name_pgf = str(plot_file_name) + ".pgf"
    print(plot_file_name_pgf)
            
    gs = gsg.graph_summary_generator( gs = cfg.getint('GraphSummary', 'GS-Model') )
   
    gs.load( graph_data )
        
    print("Model: [","num_features:", gs.get_num_features(), "num_classes:",gs.get_num_classes(),"]")
    
    max_classes = cfg.getint('WorkParameter', 'max_classes')
    
    summary_path = run_dir / str(cfg.getint('GraphSummary', 'GS-Model')).zfill(2)
    print("Graphsummary path:", summary_path)
    pathlib.Path( summary_path ).mkdir(parents=True, exist_ok=True)
    gs.plot_class_distribution( filename = str( summary_path / plot_file_name_png), max_classes = max_classes )
    gs.plot_class_distribution( filename = str( summary_path / plot_file_name_pgf), max_classes = max_classes )
    
    
if __name__ == "__main__":
    main()

