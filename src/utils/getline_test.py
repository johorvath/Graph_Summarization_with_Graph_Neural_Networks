"""
getline_test.py: test our own line reader compared to rdflib
"""

import argparse
import rdflib
import rdflib.parser as rdflib_parser
import rdflib.plugins.parsers.ntriples as triples
import rdflib.plugins.parsers.nquads as quads

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,dest='path', help='path to config file')  
args = parser.parse_args()


subjects = []
objects = []
predicates = []

# Using readline()
file1 = open(args.path, 'r')
count_line = 0
count_invalid = 0
count_edge = 0

while True:
    # Get next line from file
    line = file1.readline()
 
    # if line is empty
    # end of file is reached
    if not line:
        break
    
    count_line += 1
        
    #print("Line{}: {}".format(count_line, line.strip()))
    sink = rdflib.ConjunctiveGraph()
    parseq = quads.NQuadsParser()
    strSource = rdflib_parser.StringInputSource(line.encode('utf-8'))
    
    try:
        #try parsing the line to a valid N-Quad
        parseq.parse(strSource, sink)
        count_edge += 1
        
        #write the validated N-Quad into the filtered File

        subjects.append(list(sink.subjects())[0])
        predicates.append(list(sink.predicates())[0])
        objects.append(list(sink.objects())[0])
                                        
    
    except triples.ParseError:
        #catch ParseErrors and write the invalidated N-Quad into the trashed File
        count_invalid += 1
        
        #print the number of Errors and current trashed line to console
        #print('Wrong Line Number ' + str(f'{count_invalid:,}') + ': ' + line)
    
print("lines read:", count_line)
 
file1.close()

unique_subs = set(subjects )
unique_objs = set(objects )
unique_nodes = list( unique_subs.union( unique_objs ) )
unique_nodes = [ str(s) for s in unique_nodes ]
num_nodes_ = len( unique_nodes )
num_features_ = len( set( predicates ) )

ind = ["http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in x for x in predicates]
print("rdf occurences: ", np.array(ind).sum() )


print("subjects:",len(subjects))
print("objects:",len(objects))
print("predicates:",len(predicates))
print("unique_subs:",len(unique_subs))
print("unique_objs:",len(unique_objs))
print("vertices:",len(unique_nodes))
print("num_features_:",num_features_)
