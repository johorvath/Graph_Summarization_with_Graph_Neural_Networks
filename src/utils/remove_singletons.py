"""
remove_singletons.py: remove singletons from the data distribution, update class labels of
the subgraphs into a new file and update the description file
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

import pandas as pd

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()
        
    cfg = configparser.ConfigParser()
    print(args)
    cfg.read(args.config)
    
    base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'])
    run_dir = base_dir / cfg_u.makePath(cfg['DataExchange']['run_dir'])
        
    folds =  ast.literal_eval(cfg.get('WorkParameter', 'folds'))
    
    desc_file = cfg.get( 'WorkParameter','desc_file' )
    min_support = cfg.getint( 'WorkParameter','min_support' )
    
    data_folds = utils.load_data_fold( run_dir, folds, str(desc_file + ".csv") )
    
    print(data_folds)
    
    classes = data_folds['class'].tolist()
    unique_class = list( set ( classes  ) )
    
    count_di = {x:  classes.count( x ) for x in unique_class }
    
    count = len(list(data_folds.iterrows()))
    
    print("Total number of subgraph:", count)
        
    print(count_di)    
    print("Number of classes before removal of singleton: ", len(count_di) )
    
    no_singleton_di = {key: value for key, value in count_di.items() if value > min_support }
    print(no_singleton_di)
    print("Number of classes after removal of singleton: ", len(no_singleton_di) )
    
    label_list = set(no_singleton_di.keys())
    
    if True:                
        drop_counter = 0
        #drop singleton occurences from the description file and save new one
        for f in folds:
            desc_file1 = run_dir / str(f).zfill(3) / str( str(desc_file) + ".csv" )
            print(desc_file1)
            df = pd.read_csv(desc_file1)
            do = df.copy()
            counter = 0
            for index, row in df.iterrows():
                if( row['class'] not in label_list ):
                    do = do.drop([index])
                    counter+=1                
            drop_counter += counter
            desc_file2 = run_dir / str(f).zfill(3) /str( str(desc_file) + "_min_support_" + str(min_support) + ".csv" )
            do.to_csv(str( desc_file2 ), index=False, encoding='utf8')
            print("fold ", f, " dropped ", counter, " vertices/singleton classes.")    
        print(drop_counter)
        
    

if __name__ == "__main__":
    main()