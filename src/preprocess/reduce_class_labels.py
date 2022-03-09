"""
reduce_class_labels.py: load the panda dataframe from the description files, 
update the class labels to the new label range and save the changes inplace
"""
import argparse
import configparser
import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

from neural_model import utils

import pandas as pd
import torch

from config_utils import config_util as cfg_u

import ast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    torch.set_num_threads(cfg.getint( 'WorkParameter','num_threads' ))
    
    base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'])
    run_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['rundir'])
    
    summary_path = run_dir / str(cfg.getint('GraphSummary', 'GS-Model')).zfill(2)
    print("Graphsummary path:", summary_path)
    
    fold_list =  ast.literal_eval(cfg.get('WorkParameter', 'fold_list'))    
    
    desc_file = cfg.get( 'GraphSummary','desc_file' )
    
    data_folds = utils.load_data_fold( summary_path, fold_list, desc_file )    
    print(data_folds)
           
    classes = data_folds['class'].tolist()
    unique_class = list( set ( classes  ) )
    
    count_di = {x:  classes.count( x ) for x in unique_class }
    print(count_di)
    
    count_single = len(list(data_folds.iterrows()))
    
    print("Total number:", count_single)
    
    print("Unique classes:", unique_class )
    
    print( "Classes were reduced from", max( classes ), "to", len( unique_class ) )
    
    for i in fold_list:
        print("Fold",i)
        desc_file_fold = summary_path / str(i).zfill(3) / desc_file
        print( "Load description file:", desc_file_fold )
        df = pd.read_csv(desc_file_fold)    
        print("b",df)
        for index, row in df.iterrows():
            df.loc[index,"weight"] = int(( 1 - ( count_di[row['class']] / count_single ) ) * 100) 
            df.loc[index,"class"] = unique_class.index( row['class'] ) + 1           
            
        df.to_csv(str( desc_file_fold ), index=False, encoding='utf8')
        print("a",df)        
    

if __name__ == "__main__":
    main()