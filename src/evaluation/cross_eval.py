# -*- coding: utf-8 -*-
"""
cross_eval.py: calculate mean and std_dev over the values given in argument --file
"""

import argparse

import pandas as pd

import numpy as np

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,dest='file', help='path to k-cross csv file')  
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    acc = np.array( df["test_acc"] )
    
    print(args.file, "[mean:", np.mean(acc, axis=0), "std_error:", np.std(acc, axis=0)/np.sqrt(len(acc)),"]")
    
    
if __name__ == "__main__":
    main()