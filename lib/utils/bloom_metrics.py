"""
bloom_metrics.py: utility functions to calculate the metrics for the Bloom filter
"""

from itertools import dropwhile
import numpy as np


def gini_index(p):
    """
    Calculates the gini_index over given probabilities

    
    Args:
        p (np.array): probabilities
    """
    return 1 - np.sum(np.power(p,2))

def imp_acc(gs, cl):
    """
    Calculate the accuracy and impurity in the given graph summary; 
    assumption: vertices in the biggest ground truth clusters per bloom filter hash cluster
    are the true positives

    
    Args:
        gs (dict): graph information dictionary
        cl (dict): dictionary of clustered ground truth vertices per bloom filter hash
    """
    impurity = []
    tp = 0.0
    s = 0.0
    for key, value in cl.items():
        p = np.array([ value.count(x) for x in set(value) ])
        
        tp += p.max()
        s += p.sum()
        
        p = p / len(value)
        imp = ( len(value) / len(gs.graph_information_) ) * gini_index(p)
        impurity.append(imp)
    print("tp: ", tp, " total_count: ", s)
    return impurity, ( tp / s )

def rand_ind( gs ):
    """
    Calculate the rand index
    
    Args:
        gs (dict): graph information dictionary
    """
    sorted_gs = sorted( gs.graph_information_ )
    ss = 0.0
    dd = 0.0
    sd = 0.0
    ds = 0.0
    for s1 in sorted_gs:
        info1 = gs.graph_information_[s1]
        # skip already evaluated edges
        for s2 in dropwhile(lambda v: v <= s1, sorted_gs):    
            info2 = gs.graph_information_[s2]
            if ( info1.hash == info2.hash and info1.bloomfilter == info2.bloomfilter ):
                ss += 1
            if ( info1.hash != info2.hash and info1.bloomfilter != info2.bloomfilter ):
                dd += 1
            if ( info1.hash == info2.hash and info1.bloomfilter != info2.bloomfilter ):
                # should not trigger
                sd += 1
            if ( info1.hash != info2.hash and info1.bloomfilter == info2.bloomfilter ):
                ds += 1
    print("ss: ", ss, " dd: ", dd, " sd: ", sd, " ds: ", ds )
        
    return ( (ss+dd)/(ss+dd+sd+ds) )