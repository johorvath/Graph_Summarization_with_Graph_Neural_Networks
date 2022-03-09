"""
file_filter.py: Reads nquads in rdf format, validates and filter them.
"""

import site
site.addsitedir('../../lib')  # Always appends to end

import re
import gzip

import rdflib
import rdflib.parser as parser
import rdflib.plugins.parsers.ntriples as triples
import rdflib.plugins.parsers.nquads as quads

from config_utils import config_util as cfg_u


def filter( fileDirectoryAndName, writeFileName, trashFileName, preSkolemize=False, numLines=0 ):
    """
    This function filters all non rdf conform lines from an input file and writes a filtered and trashed file

    
    Args:
        fileDirectoryAndName (pathlib.PurePath): Path to nquads file
        writeFileName (pathlib.PurePath): Path for the validated and filtered data
        trashFileName (pathlib.PurePath): Path for the filtered trash
        preSkolemize (bool): preSkolemize is a Boolean which replaces all blanknodes with handmade IRIs on True ( default is False )
        numLines (int): numLines is the number of lines which shall be checked in the input file, numLines == 0 sets the number to the the number of lines in the file ( default in 0 )
    """
    
    readFile = gzip.open(fileDirectoryAndName, 'rt', encoding = 'utf-8') #gzip needs specified text mode ('t') if encoding gets used
    writeFile = gzip.open(writeFileName, 'wt+', encoding ='utf-8')
    trashFile = gzip.open(trashFileName, 'wt+', encoding ='utf-8')
    
    #set the number of lines to the number of lines in the file
    if numLines == 0:
        numLines = sum(1 for line in gzip.open(cfg_u.makePath(fileDirectoryAndName), 'rt', encoding = 'utf-8'))
    
    #generate the sink which is used to test the quads
    sink = rdflib.ConjunctiveGraph()
    errNo = 0
    ignoreNo = 0
    
    for i in range(numLines):        
        #progress output
        if( i%100000 == 0 ):
            print(str(f'{i:,}') + " of " + str(f'{numLines:,}') + " lines.")            
        
        line = readFile.readline()
        
        if preSkolemize:
            
            #splitting at non escaped " (preparation for preskolemization) 
            quotationSplitLine = re.split(r'(?<!\\)\"', line) #(negative lookbehind regex in order to respect escaped " in literals)
            
            #handle blanknodes
            if '_:' in line:
                if(len(quotationSplitLine) == 1): #no literal as object in line
                    line = manageBNode_(line)
                elif (len(quotationSplitLine) == 3): #literal as object in line
                    line = ''
                    for j in range(0, len(quotationSplitLine)):
                        if(j%2 == 1): #literals
                            line = line + '"' + quotationSplitLine[j] + '"'
                        else: #non-literals
                            line = line + manageBNode_(quotationSplitLine[j])
                else:
                    #failed to eat line with literal, therefore emptying it on order to ignore it completely
                    ignoreNo += 1
                    print('Ignored Line Number ' + str(f'{ignoreNo:,}') + ': ' + line)
                    line = ''
                    
        parseq = quads.NQuadsParser()
        strSource = parser.StringInputSource(line.encode('utf-8'))
        
        try:
            #try parsing the line to a valid N-Quad
            parseq.parse(strSource, sink)
            
            #write the validated N-Quad into the filtered File
            writeFile.write(line)
        
        except triples.ParseError:
            #catch ParseErrors and write the invalidated N-Quad into the trashed File
            trashFile.write(line)
            
            #print the number of Errors and current trashed line to console
            errNo += 1
            print('Wrong Line Number ' + str(f'{errNo:,}') + ': ' + line)
    
    
    #close all Filereaders
    readFile.close()
    writeFile.close()
    trashFile.close()
    
    if (ignoreNo > 0):
        print('Total Number of wrong N-Quads: ' + str(f'{errNo:,}') + ' and total Number of ignored N-Quads: ' + str(f'{ignoreNo:,}'))
    else:
        print('Total Number of wrong N-Quads: ' + str(f'{errNo:,}'))


def toGzip(fileDirectoryAndNameToGzip):
    """
    This function transform a file to gzip

    
    Args:
        fileDirectoryAndNameToGzip (pathlib.PurePath): Path to the file
    """
    
    file = open(cfg_u.makePath(fileDirectoryAndNameToGzip), 'rb')
    fileGzipped = gzip.open(cfg_u.makePath(str(fileDirectoryAndNameToGzip) + '.gz'), 'wb+')
    fileGzipped.writelines(file)
    
    fileGzipped.close()
    file.close()


def manageBNode_(line):    
    """
    This function replaces all blanknodes in the line with a handmade IRI (prepend 'https://blanknode/', delete the '_:' and append '>')

    
    Args:
        line (str): Line to manage
    """
    line = re.sub('( |^)_:', '\g<1><https://blanknode/', line)
    line = re.sub("(( |^)[^@ ]*[^^,>\" ]) ", "\g<1>> ", line)
    
    return line