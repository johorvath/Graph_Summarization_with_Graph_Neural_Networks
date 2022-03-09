"""
utils.py: Wrapper functions for the modules torch and pandas.
"""

import torch
import torch.nn.functional as F
import torch_geometric

import timeit

import pandas as pd


def calc_acc( logits, y ):
    """
    Calculates the needed data to train the networks

    
    Args:
        logits (list): Predicted class labels
        y (list): Target class labels
    """
    #get prediction
    pred = logits.max(1)[1]
    return pred.eq(y).sum().item() / y.size()[0]
    
def load_fold( path, fold_list, load_data, dsc_file = "description.csv" ):
    """
    Load the given folds and concatenate the description files

    
    Args:
        path (pathlib.PurePath): Path to folds
        fold_list (list): List of folds we want to use
        dsc_file (str): name of the description file
    """
    df_concat = pd.DataFrame({'file':[],'degree':[],'weight':[]})
    
    print("Start convert into dict!")
    data_dict = dict(zip(load_data.file, load_data.file_path))
    print("Finish convert into dict!")
    
    for i in fold_list:
        desc_file = path / str(i).zfill(3) / dsc_file
        df = pd.read_csv(desc_file)
        #print(df)
        df['file_path'] = df['file'].apply(lambda x: data_dict[x] )     
        df_concat = pd.concat([df_concat,df], axis=0, ignore_index=True)
    return df_concat

def load_data_fold( path, fold_list, dsc_file = "description.csv" ):
    """
    Load the given folds and concatenate the description files

    
    Args:
        path (pathlib.PurePath): Path to folds
        fold_list (list): List of folds we want to use
        dsc_file (str): name of the description file
    """
    df_concat = pd.DataFrame({'file':[],'degree':[],'weight':[]})

    for i in fold_list:
        desc_file = path / str(i).zfill(3) / dsc_file
        df = pd.read_csv(desc_file)
        df['file_path'] = df['file'].apply(lambda x: str( path / str(i).zfill(3) / x ) )      
        df_concat = pd.concat([df_concat,df], axis=0, ignore_index=True)
    return df_concat

def stichted_training( device, data_list, model, optimizer, writer
               , checkpoint_file, epochs = 200, guard_condition = 50, val_sample_size = 0, val_step = 1 ):
    """
    Execute mini batch based training

    
    Args:
        device (str): Cuda device
        data_list (dict): Dictionary of training and validation subgraphs
        model (torch_geometric.nn.Module): The chosen GNN-model
        optimizer (torch.optim.Optimizer): The chosen optimizer
        writer (torch.utils.tensorboard.writer.SummaryWriter): Summary writer for tensorboard
        checkpoint_file (torch.utils.checkpoint.Checkpoint): Checkpoint file object
        epochs (int): Number of epochs ( default is 200 )
        guard_condition(int): Mini batch size ( default is 50 )
        val_sample_size (int): Number of validation mini batches ( default is 0 )
        val_step (int): Step width for validation ( default is 1 )
    """
    start_train = timeit.default_timer()
    for epoch in range(epochs):        
        start_ep = timeit.default_timer()
        data = draw_samples( data_list['train'], guard_condition)        
        data = data.to(device)
       
        model.train()
        optimizer.zero_grad()
        out = model(data)
        
        #filter out transductive and dummy vertices
        mask = [data.y > 0]        
        
        loss = F.nll_loss(out[mask], data.y[mask])
        
        loss.backward()
        optimizer.step()

        train_acc = calc_acc( out, data.y )        
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        del data
        
        #possibility to run validation ever val_step to reduce runtime
        if epoch % val_step == 0:
            val_acc = evaluate_model( model, data_list['val'], device, val_sample_size, guard_condition)
            writer.add_scalar('Accuracy/val', val_acc, epoch)                                
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
            print(log.format(epoch, train_acc, val_acc))
        else:
            log = 'Epoch: {:03d}, Train: {:.4f}'
            print(log.format(epoch, train_acc))
        
        writer.add_scalars('Accuracy', {"train": train_acc, "val" : val_acc }, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, checkpoint_file )
        
        end_ep = timeit.default_timer()
        overall_timediff = end_ep - start_train
        print('Epoch time: {:.02f}; Time overall {:.02f}; ETA {:.02f}'.
                  format(end_ep - start_ep, overall_timediff
                         , ( overall_timediff / ( epoch + 1 ) ) * ( len(data_list['train']) - ( epoch + 1 ) ) ) )

    
def test_model( model, data_list, device, writer, test_sample_size, guard_condition ):    
    """
    Test model

    
    Args:
        model (torch_geometric.nn.Module): The chosen GNN-model
        data_list (dict): Dictionary of training and validation subgraphs
        device (str): Cuda device
        writer (torch.utils.tensorboard.writer.SummaryWriter): Summary writer for tensorboard
        test_sample_size (int): Number of test mini batches
        guard_condition(int): Mini batch size
    """
    print("--------------")        
    print("Run test now!")
    start_ep = timeit.default_timer()
    test_acc = evaluate_model( model, data_list, device, test_sample_size, guard_condition )
    writer.add_scalar('Accuracy/test', test_acc, -1 )
    log = 'Epochs: {:03d}, Test: {:.4f}'
    print(log.format(-1, test_acc))
    end_ep = timeit.default_timer()
    print('Test time: {:.02f}s for {:03d} batches'.
          format( end_ep - start_ep, test_sample_size ) )
    return test_acc

@torch.no_grad()
def evaluate_model( model, data_list, device, sample_size, guard_condition ):
    """
    Evaluate model

    
    Args:
        model (torch_geometric.nn.Module): The chosen GNN-model
        data_list (dict): Dictionary of training and validation subgraphs
        device (str): Cuda device
        sample_size (int): Number of mini batches
        guard_condition (int): Mini batch size
    """
    model.eval()
    accs = 0
    #calculate the average over multipe inferences
    for i in range(sample_size):       
        data = draw_samples( data_list, guard_condition )        
        data = data.to(device)
        
        logits = model(data)   
        
        #filter out transductive and dummy vertices
        mask = [data.y > 0]   
        
        accs  += calc_acc( logits[mask], data.y[mask] )
        del data
    return ( accs / sample_size )
        
      
def dummy_element_process_impl( num_features, data ):
    """
    Create dummy vertex

    
    Args:
        num_features (int): Number of features
        data (torch_geometric.data.Data): Mini batch
    """
    data.x = torch.cat( ( data.x, torch.zeros( 1, num_features ) ), 0 )
    data.y = torch.cat( ( data.y, torch.neg( torch.ones( 1, dtype=torch.long ) ) ), 0 )    
    
def draw_samples( fold_data, guard_condition ):
    """
    Draw subgraph samples for mini batch

    
    Args:
        fold_data (pandas.DataFrame): Possible list of subgraphs
        guard_condition (int): Mini batch size
    """
    x = torch.tensor( () )
    y = torch.tensor( (), dtype=torch.long )
    edge_index = torch.tensor( [ [], [] ], dtype=torch.long )
    edge_attr = torch.tensor( (), dtype=torch.long )
    
    # create pytorch data object
    data = torch_geometric.data.Data( x=x,edge_index=edge_index,y=y, edge_attr = edge_attr ) 

    while( data.x.shape[0] <  guard_condition ):
        s = fold_data.sample(n=1, weights='weight')
        # Degree of subgraph too big
        #+1 because of root node
        if ( data.x.shape[0] + s['degree'].values[0] + 1 > guard_condition ):
            if( data.x.shape[0] == 0 ):
                print("Degree exploded")
                continue
            else:
                print("Add",guard_condition - data.x.shape[0],"dummy elements")
                # fill mini-batch with dummy elements to the guard_conditionsize
                for i in range( guard_condition - data.x.shape[0] ):
                    dummy_element_process_impl( data.x.shape[1], data )
                break
            
        # Subgraph can be added
        f = s['file_path'].values[0]
        d = torch.load(f)  
        d_y = torch.zeros( d.x.shape[0], dtype=torch.long )
        d_y[0] = s['class'].values[0]
        # offset index to keep consistent indices on the mini-batch
        index_offset = data.x.shape[0]
        data.x = torch.cat( ( data.x, d.x ), 0 ) 
        data.y = torch.cat( ( data.y, d_y ), 0 )
        data.edge_attr = torch.cat( ( data.edge_attr, d.edge_attr ), 0 ) 
        d.edge_index += index_offset
        data.edge_index = torch.cat( ( data.edge_index, d.edge_index ), 1 )  
    print(data)
    return data
        

def Ncontrast(x_dis, adj_label, tau = 1):
    """
    Compute the Ncontrast loss
    Functions for GraphMLP see https://github.com/yanghu819/Graph-MLP

    
    Args:
        x_dis (torch.Tensor): Intermediate result layer
        adj_label (torch.Tensor): Adjacency matrix 
        tau (int): Temperature ( default is 1 )
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss
    
def get_A_r(data, r):
    """
    Convert adjaceny list to matrix and precalculate adjacency matrix based on k-hop

    
    Args:
        data (torch_geometric.data.Data): Mini batch
        r (int): K-hop
    """
    adj = torch.sparse.FloatTensor(
            data.edge_index, 
            torch.ones(data.edge_index.shape[1]), 
            [data.x.shape[0],data.x.shape[0]])
    adj_m = adj.to_dense()
    adj_m[adj_m!=0]=1.0
    adj_label = adj_m.fill_diagonal_(1.0)
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label@adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label

