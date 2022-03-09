"""
graph_info.py: load all description.csvs from the given list and calculate the
mean,median,given percentiles and sum of degrees
"""

import argparse
import configparser
import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

from neural_model import utils

from config_utils import config_util as cfg_u

import ast

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
    
    folds =  ast.literal_eval(cfg.get('WorkParameter', 'folds'))
    
    percentile =  ast.literal_eval(cfg.get('WorkParameter', 'percentile'))
    
    desc_file = cfg.get( 'WorkParameter','desc_file' )
    data_col = cfg.get( 'WorkParameter','data_col' )
    
    data_folds = utils.load_data_fold( run_dir, folds, desc_file )
    
    print(data_folds)
    
    data1 =  np.array(data_folds[data_col].tolist())

    mean = np.mean(data1)
    median = np.median(data1)
    stdev = np.std(data1)
    perc = {x: np.percentile(data1, x) for x in percentile}
    sum1 = np.sum(data1)
    
    # Printing the mean
    print("Data mean is :", mean, "+-", stdev, "median:", median, "percentile:",perc, "sum:", sum1)
    
    unique, counts = np.unique(data1, return_counts=True)
    
    mean = np.mean(counts)
    median = np.median(counts)
    stdev = np.std(counts)
    perc = {x: np.percentile(counts, x) for x in percentile}
    sum1 = np.sum(counts)
    
    print("Histogram mean is :", mean, "+-", stdev, "median:", median, "percentile:",perc, "sum:", sum1)
        

if __name__ == "__main__":
    main()
