"""
rdfgraph_information.py: load the rdf-graph and count the occurrences for a given regex 
on the predicates
"""
import argparse
import configparser
import site
import sys
import fnmatch

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

import rdflib
import gzip

from config_utils import config_util as cfg_u

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'])
    
    filtered_datafile  =  base_dir / cfg_u.makePath(cfg['Dyldo']['filtered_datafile'])
    
    regex = cfg['WorkParameter']['filter_match']
        
    #load graph
    g = rdflib.ConjunctiveGraph()
    rdfdata = gzip.open(filtered_datafile, "rb")
    g.parse(rdfdata, format="nquads")
    rdfdata.close()
    print("Graph with " + str(len(g)) + " Elements has been created")
    
    #extract predicates
    pred = list(g.predicates())
    pred = [str(p) for p in pred]
    print(pred)
    
    #apply regex filter and output results
    print("Results for: ", regex)    
    filtered = fnmatch.filter(pred, regex)    
    print("Number of total occurrences: ", len(filtered))
    filtered_unique = fnmatch.filter(set(pred), regex)    
    print("Number of unique occurrences: ", len(filtered_unique))
        


if __name__ == "__main__":
    main()