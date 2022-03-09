"""
run_preprocessing_schemex.py: run complete preprocessing pipeline of RDF-filter, 
graphsummary calculation and information generation and subgraph concatenation based 
on preexisting subgraphs of an 1-hop model
"""

import argparse
import configparser
import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

import torch

from neural_model import utils

from config_utils import config_util as cfg_u

from dyldo_rdflib import file_filter as ff

from graph_summary_generator import graph_summary_generator as gsg

import pathlib

import ast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    torch.set_num_threads(cfg.getint( 'WorkParameter','num_threads' ))
    
    load_data = cfg.getboolean('WorkParameter', 'load_data')
    run_dyldo_filter = cfg.getboolean('WorkParameter', 'run_dyldo_filter')
    calc_gs = cfg.getboolean('WorkParameter', 'calc_graph_summary')
    fold_list =  ast.literal_eval(cfg.get('WorkParameter', 'fold_list'))
    save_fold_percentage = cfg.getfloat('WorkParameter', 'save_fold_percentage')
    edge_sampling_percentage = cfg.getfloat('WorkParameter', 'edge_sampling_percentage')
    min_support_classes =  cfg.getint( 'WorkParameter','min_support_classes' )
    mini_batch_size =  cfg.getint( 'WorkParameter','mini_batch_size' )
        
    base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'])
    run_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['rundir'])
    load_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['loaddir'])
    
    raw_datafile       =  base_dir.joinpath( cfg_u.makePath(cfg['Dyldo']['raw_datafile']) )
    filtered_datafile  =  base_dir / cfg_u.makePath(cfg['Dyldo']['filtered_datafile'])
    trashed_datafile   =  base_dir / cfg_u.makePath(cfg['Dyldo']['trashed_datafile'])
    numLines =  cfg.getint('Dyldo', 'num_lines')
    preSkolemize =  cfg.getboolean('Dyldo', 'pre_skolemize')
    
    graph_data =  base_dir / cfg_u.makePath(cfg['GraphSummary']['save_file'])    
    
    k_folds =  cfg.getint( 'GraphSummary', 'k_folds' )
            
    max_items =  cfg.getint( 'GraphSummary','max_items' )
    error_rate =  cfg.getfloat( 'GraphSummary','error_rate' )
            
    
    extract_subgraph = cfg.getboolean( 'GraphSummary','extract_subgraph' )
    desc_file = cfg.get( 'GraphSummary','desc_file' )
    load_desc_file = cfg.get( 'GraphSummary','load_desc_file' )
    
    gs = gsg.graph_summary_generator( gs = cfg.getint('GraphSummary', 'GS-Model'), 
                                     max_items = max_items, error_rate= error_rate, extract_subgraph = extract_subgraph )
    
    bloomfilter =  cfg.getboolean('GraphSummary', 'bloomfilter')
    
    fold = list(range(10))
    print(load_desc_file)
    data = utils.load_data_fold( load_dir, fold, dsc_file = load_desc_file )
   
    if load_data:  
        gs.load( graph_data )
        gs.extract_subgraph = extract_subgraph
    else:
        if run_dyldo_filter:
            ff.filter( raw_datafile, filtered_datafile, trashed_datafile, preSkolemize=preSkolemize, numLines=numLines)
        
        if calc_gs:                
            gs.create_graph_information( filtered_datafile, k_folds, str(base_dir/"subject_list") )
            gs.calculate_graph_summary( bloomfilter )
            gs.save( graph_data ) 
        
    print("Model: [","num_features:", gs.get_num_features(), "num_classes:",gs.get_num_classes(),"]")
    
    summary_path = run_dir / str(cfg.getint('GraphSummary', 'GS-Model')).zfill(2)
    print("Graphsummary path:", summary_path)
    pathlib.Path( summary_path ).mkdir(parents=True, exist_ok=True)
    
    print(fold_list)
    

    
    for fold in fold_list:
        gs.save_fold_schemex( int(fold), summary_path, data, save_fold_percentage
                             , edge_sampling_percentage, min_support_classes, mini_batch_size, desc_file )

    
if __name__ == "__main__":
    main()

