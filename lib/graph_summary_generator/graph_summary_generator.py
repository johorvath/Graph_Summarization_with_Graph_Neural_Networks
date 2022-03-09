"""
graph_summary_generator.py: Generates internal data structure to work with the given graph and calculate graph summaries out of it.

* generate internal data structure as so called graph information
* save and load graph information
* generate graph summaries out of graph information and save as folds
* getter for information like features, classes and nodes
* plot class distribution
"""

import torch
import torch_geometric
import random
import pickle
import os.path as osp

import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

import timeit

import pathlib

import pandas as pd

from bloom_filter2 import BloomFilter

from utils.plot_class_distribution import plot_class_distribution

import rdflib
import rdflib.parser as rdflib_parser
import rdflib.plugins.parsers.ntriples as triples
import rdflib.plugins.parsers.nquads as quads

import math


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj (name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

class subject_information():
    """
    A class used to represent subject informations
    """
    def __init__( self , p, o, index ):
        """
        Initialize the subject_information object

        
        Args:
            p (str): First predicate of the subject
            o (str): First object of the subject
            index (int): Index of the subject
        """
        self.hash = 0
        self.edges = [(p, o)]
        self.weight = 0
        self.fold = 0
        self.degree = 1
        self.index = index
        self.bloomfilter = 0

class graph_summary_generator():
    """
    A class used to calculate graph summaries
    """
    
    def __init__( self, gs = 1, max_items = 1, error_rate = 0.1, extract_subgraph = True ):
        """
        Initialize the graph_summary_generator object

        
        Args:
            gs (int): The chosen graph summary model index ( default is 1 )
        """
        self.graph_information_ = {}     # internal data structure
        self.weights_ = []               # occurence percentage of the classes
        self.label_dict_ = {}            # key: hash value: list( subjects )
        self.label_list_ = []            # unique hashes
        self.feature_list_ = []          # unique features
        self.counter_hashes = 0
        self.counter_nodes = 0
        self.gs_ = gs                    # index which graph summary is used ( 1 = attribute based collection; 2 = class based collection; 3 = schemex; 4 = property type collection )
        self.num_vertices_ = 0
        self.num_features_ = 0
        
        self.k_folds = []
        
        self.cached_subgraphs = {}
        
        self.error_rate = error_rate
        self.max_items = max_items
        
        self.extract_subgraph = extract_subgraph
        
        self.k_ = 0
        self.gs_name = ""
        if( self.gs_ == 1 ):
            print("attribute based collection")
            self.k_ = 1
            self.gs_name = "attribute based collection"
        elif( self.gs_ == 2 ):
            print("class based collection")
            self.k_ = 1 
            self.gs_name = "class based collection"
        elif( self.gs_ == 3 ):
            print("schemex")
            self.k_ = 2 
            self.gs_name = "schemex"
        elif( self.gs_ == 4 ):
            print("property type collection")
            self.k_ = 1 
            self.gs_name = "property type collection"
        else:
            print("wrong index -> no graph summary defined for " + str(self.gs_))

    def create_graph_information( self, path, k_fold, file = None ):
        """
        Calculates the needed data to train the networks

        
        Args:
            path (str): path to the nq-file
            k_fold (int): Number of folds to calculate
        """
        self.unique_subjects = []
        self.unique_subjects_set = set()
        self.subject_index = {}
        
        if file != None:      
            print("Check if file exists")
            if osp.isfile(file + ".pkl"):
                self.subject_index = load_obj( file )
                print("Loaded",len(self.subject_index), "subjects!")
        
        subjects = []
        objects = []
        predicates = []
        
        with open(path) as f:
            number_lines = sum(1 for _ in f)
        
        print("Files has",number_lines,"lines.")
        
        index = 0
        
        # Using readline()
        with open(path) as f:
            count_line = 0
            count_invalid = 0
            count_edge = 0

            
            while True:
                # Get next line from file
                line = f.readline()
             
                # if line is empty
                # end of file is reached
                if not line:
                    break
                
                count_line += 1
                
                if count_line % 10000 == 0:
                    print("Read line", count_line, "of", number_lines, "(", count_line / number_lines * 100.0,"%)")
                    
                #print("Line{}: {}".format(count_line, line.strip()))
                sink = rdflib.ConjunctiveGraph()
                parseq = quads.NQuadsParser()
                strSource = rdflib_parser.StringInputSource(line.encode('utf-8'))

                try:
                    #try parsing the line to a valid N-Quad
                    parseq.parse(strSource, sink)
                    
                    count_edge += 1
                    
                    #write the validated N-Quad into the filtered File
                    
                    #print( list(sink.subjects()),list(sink.predicates()),list(sink.objects() ) )
                    s = str(list(sink.subjects())[0])
                    p = str(list(sink.predicates())[0])
                    o = str(list(sink.objects())[0])
                    
                    if s not in self.unique_subjects_set:
                        self.unique_subjects.append(s)
                        self.unique_subjects_set.add(s)
                        index += 1
                        self.subject_index[s] = index
                    
                    subjects.append( s )    
                    predicates.append( p )
                    objects.append( o )
                    
                    if( s in self.graph_information_ ):
                        self.graph_information_[s].edges.append((p, o))
                        self.graph_information_[s].degree += 1
                    else:                                
                        self.graph_information_[s] = subject_information( p, o, self.subject_index.get(s) )   
                    
                    
                except triples.ParseError:
                    #catch ParseErrors and write the invalidated N-Quad into the trashed File
                    count_invalid += 1
                    
                    #print the number of Errors and current trashed line to console
                    #print('Wrong Line Number ' + str(f'{count_invalid:,}') + ': ' + line)
                
            print("lines read:", count_line)
            print("invalid lines read:", count_invalid)
            print("total edges:", count_edge, "/", len(subjects))
            
        unique_objs = set( objects )
        self.num_features_ = len( set( predicates ) )
        self.unique_vertices = list( set( self.unique_subjects ).union( unique_objs ) )
        self.num_vertices_ = len( self.unique_vertices )
        
        print("number features:", self.num_features_)
        print("number vertices:", self.num_vertices_)
        print("number unique subjects:", len( self.unique_subjects )) 
            
        
        if file != None:                  
                print("Store", len(self.subject_index), "subjects!")
                save_obj(self.subject_index, file)
                
        self.calculate_k_folds ( k_fold )
    
    
    def calculate_graph_summary( self, bloomfilter ):
        """
        Calculate the by the index defined graph summary on the prepared data

        
        Args:
            bloomfilter (bool): Value if we want to additionally calculate the hashes with a bloomfilter
        """
        # 2. create graph summary over whole graph
        # go through graph information and calculate labels and features
        if( self.gs_ == 1 ):
            self.attribute_based_collection_impl( bloomfilter )
        elif( self.gs_ == 2 ):
            self.class_based_collection_impl( bloomfilter )  
        elif( self.gs_ == 3 ):
            self.schemex_impl( bloomfilter )
        elif( self.gs_ == 4 ):
            self.property_type_collection_impl( bloomfilter )
            
        self.num_features = len( self.feature_list_ )
        
        # create label_list
        self.label_list_ = list( self.label_dict_.keys() )
        
        # create weights for graph_informations
        for element in self.graph_information_.items():
            # calculcate weight and invert because we want an evenly distributed sample pool
            element[1].weight = int(( 1 - ( len( self.label_dict_[ element[1].hash ] ) / self.num_vertices_ ) ) * 100)
        self.weights_ = [ element[1].weight for element in self.graph_information_.items() ]
    
    def calculate_weight ( self, node_list, min_support = 0 ):
        """
        Calculate the weights of the classes in regard to the number of nodes

        
        Args:
            node_list (list): List of nodes

        Returns:
            list: A list of the calculated weights
        """
        weights = []
        num_vertices = 0
        if min_support > 0:
            for k,v in self.label_dict_:
                if ( len(v) > min_support):
                    num_vertices +=  len(v)              
        else:
            num_vertices = self.num_vertices_
            
        
        for element in node_list:
            # calculcate weight and invert because we want an evenly distributed sample pool
            weights.append ( int(( 1 - ( len( self.label_dict_[ self.graph_information_[element].hash ] ) / num_vertices ) ) * 100) )        
        return weights
        
    def add_to_label_dict( self, tmp_hash, s ):
        """
        Add to label_dict with tmp_hash keys and list of subjects values

        
        Args:
            tmp_hash (int): Calculated hash for the subject
            s (str): Subject name
        """
        if tmp_hash in self.label_dict_:
            self.label_dict_[tmp_hash].append(s)
        else:
            self.label_dict_[tmp_hash] = [s]
        
    def is_rdf_type( self, s ):
        """
        Check if the given string contains rdf and therefore is an rdf-type

        
        Args:
            s (str): Feature used by the graph summary model

        Returns:
            bool: If it is a rdf-type
        """
        return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in s
        
    def based_collection_impl( self, prop, typ, bloomfilter ):
        """
        Base implementation for the graph summary calculation

        
        Args:
            prop (bool): If we want to use the property sets
            typ (bool): If we want to use the type sets
            bloomfilter (bool): Value if we want to additionally calculate the hashes with a bloomfilter
        """
        # iterate through all subjects
        for gi in self.graph_information_.items():
            # calculate hash
            tmp_feature_list = []
            # go through edges of gi            
            for h in gi[1].edges:
                use_feature = False
                feature = h[0]
                # check if we want to use the feature in this config
                if( prop == True and self.is_rdf_type( feature ) == False ):
                    use_feature = True
                
                if( typ == True and self.is_rdf_type( feature ) == True ):
                    use_feature = True
                
                if( use_feature == True ):                        
                    # append feature for hash calculation
                    tmp_feature_list.append( feature )
                    
                # add predicate to features if not already in there
                if( feature not in self.feature_list_ ):
                    self.feature_list_.append( feature )
            
            tmp_hash_bloomfilter = 0
            if( bloomfilter ):
                bloom = BloomFilter( max_elements=self.max_items, error_rate=self.error_rate )
                for tmp_feature in tmp_feature_list:
                    bloom.add( tmp_feature )
                tmp_hash_bloomfilter = hash( str(bloom.backend.array_) )
                bloom.close()
                
            tmp_feature_list.sort()
            tmp_feature_list_string = "".join( tmp_feature_list )
            tmp_hash = hash( tmp_feature_list_string )
            
            gi[1].hash = tmp_hash
            gi[1].bloomfilter = tmp_hash_bloomfilter
            self.add_to_label_dict( tmp_hash, gi[0] )
    
    def attribute_based_collection_impl( self, bloomfilter ):
        """
        Implementation to calculate the attribute based collection

        
        Args:
            bloomfilter (bool): Value if we want to additionally calculate the hashes with a bloomfilter
        """
        self.based_collection_impl( True, False, bloomfilter )
    
    def class_based_collection_impl( self, bloomfilter ):
        """
        Implementation to calculate the class based collection

        
        Args:
            bloomfilter (bool): Value if we want to additionally calculate the hashes with a bloomfilter
        """
        self.based_collection_impl( False, True, bloomfilter )
    
    def property_type_collection_impl( self, bloomfilter ):
        """
        Implementation to calculate the property type collection

        
        Args:
            bloomfilter (bool): Value if we want to additionally calculate the hashes with a bloomfilter
        """
        self.based_collection_impl( True, True, bloomfilter )
    
    def schemex_impl( self, bloomfilter ):
        """
        Implementation to calculate schemex

        
        Args:
            bloomfilter (bool): Value if we want to additionally calculate the hashes with a bloomfilter
        """
        # iterate through all subjects
        for gi in self.graph_information_.items():
            #gi[0] is subject; gi[1] is subject_information data struct ( hash, edges )
            # calculate hash
            tmp_feature_list = []
            # go through edges of gi
            for h in gi[1].edges:
                #h[0] is edge; h[1] is object
                pfeature = h[0]
                                       
                # append feature for hash calculation; use type and property set
                tmp_feature_list.append( pfeature )
                    
                # add predicate to features if not already in there
                if( pfeature not in self.feature_list_ ):
                    self.feature_list_.append( pfeature )
                    
                # do type set of objects connected over property sets
                if( self.is_rdf_type( pfeature ) == False and h[1] in self.graph_information_.keys() ):
                    for e in self.graph_information_[h[1]].edges:
                        feature = e[0]
                        if( self.is_rdf_type( feature ) == True ):
                            # append feature for hash calculation; use type and property set
                            # also append subject onto feature to fix same hashes bug
                            tmp_feature_list.append( pfeature + feature )
            
            tmp_hash_bloomfilter = 0
            if( bloomfilter ):
                bloom = BloomFilter( max_elements=self.max_items, error_rate=self.error_rate )
                for tmp_feature in tmp_feature_list:
                    bloom.add( tmp_feature )
                tmp_hash_bloomfilter = hash( str(bloom.backend.array_) )
                bloom.close()
                
            tmp_feature_list.sort()
            tmp_feature_list_string = "".join( tmp_feature_list )
            tmp_hash = hash( tmp_feature_list_string )
            
            gi[1].hash = tmp_hash
            gi[1].bloomfilter = tmp_hash_bloomfilter
            self.add_to_label_dict( tmp_hash, gi[0] )
    
    def element_process_impl( self, element, id_list, num_features, data ):
        """
        Implementation to process the data as pytorch geometric data object

        
        Args:
            element (tuple): Tuple of subject and subject information
            id_list (list): List of unique subjects
            num_features (int): Number of overall features
            data (torch_geometric.data): Torch geometric data object

        Returns:
            torch_geometric.data: cache object for the current subgraph
            int: Number of the edges
        """
        data_cache = self.create_default_data() 
        cache_id_list = []
                
        if element[0] not in id_list:
            id_list.append( element[0] )
            data.x = torch.cat( ( data.x, torch.zeros( 1, num_features ) ), 0 )
            #data.y = torch.cat( ( data.y, torch.zeros( 1, dtype=torch.long ) ), 0 )
        
        if element[0] not in cache_id_list:            
            cache_id_list.append( element[0] )
            data_cache.x = torch.cat( ( data_cache.x, torch.zeros( 1, num_features ) ), 0 )
            #data_cache.y = torch.cat( ( data_cache.y, torch.zeros( 1, dtype=torch.long ) ), 0 )
            
        s_index = id_list.index( element[0] )
        cache_s_index = cache_id_list.index( element[0] )
        
        for i in element[1].edges:
            p_index = self.feature_list_.index( i[0] )
            
            if i[1] not in id_list:
                id_list.append( i[1] )  
                data.x = torch.cat( ( data.x, torch.zeros( 1, num_features ) ), 0 ) 
                #data.y = torch.cat( ( data.y, torch.zeros( 1, dtype=torch.long ) ), 0 )
                
            if i[1] not in cache_id_list:            
                cache_id_list.append( i[1] )  
                data_cache.x = torch.cat( ( data_cache.x, torch.zeros( 1, num_features ) ), 0 )
                #data_cache.y = torch.cat( ( data_cache.y, torch.zeros( 1, dtype=torch.long ) ), 0 )
                
            o_index = id_list.index( i[1] )
            cache_o_index = cache_id_list.index( i[1] )
            
            # set id for feature matrix
            data.x[ s_index, p_index ] = 1
            
            data_cache.x[ cache_s_index, p_index ] = 1

            # add edges to data structures
            data.edge_index = torch.cat( ( data.edge_index, torch.tensor([ [s_index], [o_index] ]) ), 1 )
            data.edge_attr = torch.cat( ( data.edge_attr, torch.tensor( [p_index] ) ), 0 )
            
            data_cache.edge_index = torch.cat( ( data_cache.edge_index, torch.tensor([ [cache_s_index], [cache_o_index] ]) ), 1 )
            data_cache.edge_attr = torch.cat( ( data_cache.edge_attr, torch.tensor( [p_index] ) ), 0 )
            
        # set hash class per node ( 1 -> n )
        #data.y[ s_index ] = self.label_list_.index( element[1].hash ) + 1
        
        #data_cache.y[ cache_s_index ] = data.y[ s_index ]
        
        return data_cache, len(element[1].edges)
        
    def void_element_process_impl( self, num_features, data ):
        """
        Implementation to process a void element as pytorch geometric data object

        
        Args:
            num_features (int): Number of overall features
            data (torch_geometric.data): Torch geometric data object
        """
        data.x = torch.cat( ( data.x, torch.zeros( 1, num_features ) ), 0 )
        #data.y = torch.cat( ( data.y, torch.neg( torch.ones( 1, dtype=torch.long ) ) ), 0 )
        
    
    def save_fold( self, fold, path, save_fold_percentage = 1, desc_file = "description.csv" ):
        """
        Implementation to save the data as fold

        
        Args:
            fold (int): Number of fold
            path (pathlib.PurePath): Path to the graph summary folder
        """
        fold_path = path / str(fold).zfill(3)
        print("Foldpath:", fold_path)
        pathlib.Path( fold_path ).mkdir(parents=True, exist_ok=True)
        d = {'file':[],'degree':[],'weight':[],'class':[]}
        start_ov = timeit.default_timer()
        
        random.seed(0)
        # sample x entries from k_folds[fold]
        new_size = int( len(self.k_folds[fold]) * save_fold_percentage ) 
        k_folds_trunc = random.sample(self.k_folds[fold], new_size )
        
        #last_invalid_index = int( ( len(k_folds_trunc) * ( 1 - save_fold_percentage ) ) )
        print("Reduce", len(self.k_folds[fold]),"to", new_size)
        
        for index, e in enumerate(k_folds_trunc):  
            start_ep = timeit.default_timer()
            element = [e, self.graph_information_[e]]
            filename = str(element[1].index).zfill( 10 ) + ".pt"
            d['file'].append(filename)
            degree = element[1].degree
            
            #only extract subgraph if needed
            #else just write the description file            
            if self.extract_subgraph:
                # create needed data structures
                x = torch.zeros( 0, self.num_features )
                edge_index = torch.tensor( [ [], [] ], dtype=torch.long )
                edge_attr = torch.tensor( (), dtype=torch.long )
               
                # create pytorch data object
                data = torch_geometric.data.Data( x=x, edge_index=edge_index, edge_attr=edge_attr ) 
            
                id_list = []   
            
                start = timeit.default_timer()
                self.element_process_impl( element, id_list, self.num_features, data )
                end = timeit.default_timer()
                print('root time: {:.02f};'.
                      format( end - start ))
                        
                print("data:", data, "degree:", degree)
                
                file = fold_path / filename
                torch.save( data, str(file) )            
                print("Saved", file)
            
            d['degree'].append( degree )
            d['class'].append( self.label_list_.index( element[1].hash ) + 1 )
            
            end_ep = timeit.default_timer()
            overall_timediff = end_ep - start_ov
            ep_diff = end_ep - start_ep
            remaining = len( k_folds_trunc )-index + 1    
            print('for-loop time: {:.02f}; Time overall {:.02f}; ETA s {:.02f}; ETA h {:.02f}; ETA d {:.02f}'.
                  format( ep_diff, overall_timediff, ( overall_timediff / ( index + 1 ) ) * remaining, ( overall_timediff / ( index + 1 ) ) * remaining / 3600, ( overall_timediff / ( index + 1 ) ) * remaining / 3600 / 24 ))
            
        d['weight'] = self.calculate_weight( k_folds_trunc )
        
        df = pd.DataFrame(data=d)
        df.to_csv(str( fold_path / desc_file ), index=False, encoding='utf8')
        
    def save_fold_schemex( self, fold, path, sub_data, save_fold_percentage = 1, edge_sampling_percentage = 1.0
                          , min_support_classes = 0,  mini_batch = 500, desc_file = "description.csv" ):
        """
        Implementation to save the data as fold

        
        Args:
            fold (int): Number of fold
            path (pathlib.PurePath): Path to the graph summary folder
            sub_data (pandas.DataFrame): Loaded fold data as concatenated description file
            save_fold_percentage (float): Percentage of data we want to use ( default is 1 )
        """
        fold_path = path / str(fold).zfill(3)
        print("Foldpath:", fold_path)
        pathlib.Path( fold_path ).mkdir(parents=True, exist_ok=True)
        d = {'file':[],'degree':[],'weight':[],'class':[]}
        start_ov = timeit.default_timer()
        
        random.seed(42)
        # sample x entries from k_folds[fold]
        new_size = int( len(self.k_folds[fold]) * save_fold_percentage ) 
        k_folds_trunc = random.sample(self.k_folds[fold], new_size )
        
        #last_invalid_index = int( ( len(k_folds_trunc) * ( 1 - save_fold_percentage ) ) )
        print("Reduce", len(self.k_folds[fold]),"to", new_size)
        
        old = len(k_folds_trunc)        
        k_folds_trunc = [e for e in k_folds_trunc if len(self.label_dict_[self.graph_information_[e].hash]) > min_support_classes]        
        print("Reduce", old,"to", len(k_folds_trunc), "because of min_support ", min_support_classes)
        
        #Filter with mini-batch size
        degree_filtered_e = []
        for index, e in enumerate(k_folds_trunc):
            element = [e, self.graph_information_[e]]
            degree = element[1].degree
            for index_j, j in enumerate(element[1].edges):                
                    if self.graph_information_.get(j[1]) != None:
                        ob_el = (j[1], self.graph_information_.get(j[1]))
                        degree += ob_el[1].degree 
            if degree < mini_batch -1:
                degree_filtered_e.append( e )
        print("Reduce", len(k_folds_trunc),"to", len(degree_filtered_e), "because of batch size ", mini_batch) 
        k_folds_trunc = degree_filtered_e
        
        count_degree = 0
          
        for index, e in enumerate(k_folds_trunc):  
            start_ep = timeit.default_timer()
            element = [e, self.graph_information_[e]]
   
            filename = str(element[1].index).zfill( 10 ) + ".pt"
            d['file'].append(filename)
            
            # create needed data structures
            x = torch.zeros( 0, self.num_features )
            #y = torch.tensor( (), dtype=torch.long )
            edge_index = torch.tensor( [ [], [] ], dtype=torch.long )
            edge_attr = torch.tensor( (), dtype=torch.long )
           
            # create pytorch data object
            data = torch_geometric.data.Data( x=x,edge_index=edge_index, edge_attr = edge_attr ) 
         
            degree = element[1].degree
            end = timeit.default_timer()
            start_n = timeit.default_timer()
            start = timeit.default_timer()
            
            edges = element[1].edges            
            old_edges = len(edges)
            edges = random.sample(edges, math.ceil( len( edges ) * edge_sampling_percentage ) )
            print("Reduce edges from", old_edges,"to", len(edges))
            
            if self.extract_subgraph:
            
                #print(degree, len( element[1].edges ) )
                
                start = timeit.default_timer()
                
                df = sub_data[sub_data['file']==filename]   
                f = df['file_path'].values[0] 
                data = torch.load(f) 
              
                print("root", data)
    
                #data.y[0] = self.label_list_.index( element[1].hash ) + 1
                end = timeit.default_timer()
                
                start_n = timeit.default_timer()
                
                
                for index_j, j in enumerate(edges):                
                    if self.graph_information_.get(j[1]) != None:
                        ob_el = (j[1], self.graph_information_.get(j[1]))
                        
            
                        degree += ob_el[1].degree 
                    
                        filename_obj = str(ob_el[1].index).zfill( 10 ) + ".pt"
                        df = sub_data[sub_data['file']==filename_obj]            
                        f = df['file_path'].values[0] 
                        data_o = torch.load(f)
                        #data_o.y = data_o.y[1:]                    
                        data_o.x = data_o.x[1:]
                        
                        index_offset = data.x.shape[0]-1
                        data.x = torch.cat( ( data.x, data_o.x ), 0 ) 
                        #data.y = torch.cat( ( data.y, data_o.y ), 0 )
                        data.edge_attr = torch.cat( ( data.edge_attr, data_o.edge_attr ), 0 ) 
                        data_o.edge_index += index_offset
                        data_o.edge_index[0,:] = index_j + 1
                        data.edge_index = torch.cat( ( data.edge_index, data_o.edge_index ), 1 )
                            
                            

                file = fold_path / filename
                torch.save( data, str(file) )            
                print("Saved", file)
            else:           
                for index_j, j in enumerate(edges):                
                    if self.graph_information_.get(j[1]) != None:
                        degree += self.graph_information_.get(j[1]).degree 
                
            print("Concated", data)            
            end_n = timeit.default_timer()
            
            d['degree'].append( degree )
            d['class'].append( self.label_list_.index( element[1].hash ) + 1 )
            
            if( degree != data.edge_attr.shape[0] ):
                count_degree += 1            
                       
            end_ep = timeit.default_timer()
            overall_timediff = end_ep - start_ov
            ep_diff = end_ep - start_ep
            remaining = len(k_folds_trunc)-index + 1    
            print('root_time: {:.02f}; hop-time: {:.02f};for-loop time: {:.02f}; Time overall {:.02f}; ETA s {:.02f}; ETA h {:.02f}; ETA d {:.02f}'.
                  format( (end-start), ( end_n -start_n ), ep_diff, overall_timediff, ( overall_timediff / ( index + 1 ) ) * remaining, ( overall_timediff / ( index + 1 ) ) * remaining / 3600, ( overall_timediff / ( index + 1 ) ) * remaining / 3600 / 24 ))
        
        d['weight'] = self.calculate_weight( list(k_folds_trunc) )
        
        print("Diff degrees:", count_degree )
        print(d)
        
        df = pd.DataFrame(data=d)
        df.to_csv(str( fold_path / desc_file ), index=False, encoding='utf8')
               
    
    def create_default_data( self ):
        """
        Create needed data structures

        Returns:
            torch_geometric.data: empty default object
        """
        x = torch.zeros( 0, self.num_features )
        #y = torch.tensor( (), dtype=torch.long )
        edge_index = torch.tensor( [ [], [] ], dtype=torch.long )
        edge_attr = torch.tensor( (), dtype=torch.long )
        data = torch_geometric.data.Data( x=x,edge_index=edge_index, edge_attr = edge_attr )         
        return data
                       

# get informations
    def get_num_features( self ):
        """
        Return number of features

        Returns:
            int: Number of features
        """
        return len( self.feature_list_ )
        
    def get_num_classes( self ):
        """
        Return number of classes

        Returns:
            int: Number of classes
        """
        return ( len( self.label_list_ ) + 1 )
    
    def get_num_vertices( self ):
        """
        Return number of vertices

        Returns:
            int: Number of vertices
        """
        return self.num_vertices_
    
    def plot_class_distribution( self, filename="class_distribution.png", max_classes = 1000 ):
        """
        Plot class distribution

        
        Args:
            filename (str): Filename ( default is "class_distribution.png" )
        """
        length_dict = {key: len(value) for key, value in self.label_dict_.items()}
        
        plot_class_distribution( length_dict, filename, max_classes = max_classes )
    
# utils
    def load(self, filename):
        """
        Load the given data file

        
        Args:
            filename (str): Filename
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close() 
        self.__dict__.update(tmp_dict) 
        self.num_features = len( self.feature_list_ )
        for i,k in enumerate(self.k_folds):
            print("Fold ", i, "size",len(k)) 

    def save(self, filename):
        """
        Save the class to a data file

        
        Args:
            filename (str): Filename
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        
    def calculate_k_folds ( self, k_fold ):
        """
        Calculate k folds

        
        Args:
            k_fold (int): Number of folds to calculate
        """
        self.k_folds = [ [] for _ in range(k_fold) ]
        nodes = list(self.graph_information_.keys())
        fold_size = int( len(nodes) / k_fold )
        print("fold_size: ", fold_size)
        start_train = timeit.default_timer()
        #create a fold possibility list
        k_folds = list(range(k_fold))
        for n in nodes:
            start_ep = timeit.default_timer()
            if not k_folds:
                self.k_folds[self.graph_information_[n].fold].append(n)    
            else:
                fold = random.choice(k_folds)
                self.graph_information_[n].fold = fold
                self.k_folds[fold].append(n) 
            
            #remove fold if full from possibility list
            if len(self.k_folds[fold]) >= fold_size and k_folds:
                k_folds.remove( fold )
            
            end_ep = timeit.default_timer()
            overall_timediff = end_ep - start_train
            #print('Epoch time: {:.02f}; Time overall {:.02f}'.
                  #format( end_ep - start_ep, overall_timediff ) )    
        for i,k in enumerate(self.k_folds):
            print("Fold ", i, "size",len(k))