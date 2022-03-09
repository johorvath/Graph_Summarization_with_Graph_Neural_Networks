import rdflib
import gzip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,dest='path', help='path to config file')  
args = parser.parse_args()

if False:
    g = rdflib.ConjunctiveGraph()
    if args.path.endswith('.nq.gz'):
        rdfdata = gzip.open(args.path, "rb")
        g.parse(rdfdata, format="nquads")
    else:
        g.parse(args.path, format="nquads")
    print("Graph with " + str(len(g)) + " Elements has been created")
    
    unique_subs = set(g.subjects())
    unique_objs = set(g.objects())
    unique_nodes = list( unique_subs.union( unique_objs ) )
    unique_nodes = [ str(s) for s in unique_nodes ]
    num_nodes_ = len( unique_nodes )
    num_features_ = len( set( g.predicates() ) )
    
    print("subjects:",len(list(g.subjects())))
    print("objects:",len(list(g.objects())))
    print("predicates:",len(list(g.predicates())))
    print("unique_subs:",len(unique_subs))
    print("unique_objs:",len(unique_objs))
    print("vertices:",len(unique_nodes))
    print("num_features_:",num_features_)


count = 0
if args.path.endswith('.nq.gz'):
    with gzip.open(args.path, "rb") as f:
        seen = set()
        for line in f:
            line_lower = line.lower()
            if line_lower in seen:
                count += 1
            else:
                seen.add(line_lower)
            
    print("Duplicate lines: ", count)
else:
    with open(args.path, "rb") as f:
        seen = set()
        for line in f:
            line_lower = line.lower()
            if line_lower in seen:
                count += 1
            else:
                seen.add(line_lower)
            
    print("Duplicate lines: ", count)
