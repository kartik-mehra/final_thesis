
from dgl.nn import SAGEConv
import dgl.function as fn
from umap.umap_ import UMAP
from sklearn.metrics import roc_auc_score
import dgl
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import json
import gzip
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import dgl.data
import os
import random
import datetime
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.options.mode.chained_assignment = None  # default='warn'

# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
  def __init__(self, in_feats, h_feats,agg_func,dropout):  # in_feats and h_feats are required parameters
    super(GraphSAGE, self).__init__()
    self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type = agg_func,feat_drop =dropout)  # Create a SAGEConv layer with specific parameters
    self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type = agg_func,feat_drop =dropout)  # Create another SAGEConv layer with specific parameters

  def forward(self, g, in_feat):
    h = self.conv1(g, in_feat)  # Apply first convolution layer
    h = F.relu(h)  # Apply ReLU activation
    h = self.conv2(g, h)  # Apply second convolution layer
    return h


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)

def train_test_split(g,test_ratio = 0.1):
    # Split edge set for training and testing
    torch.manual_seed(0)
    np.random.seed(0)
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    ## eids are the index of the dataframe representing each edge, with eids[:test_seze] we are defining that we need all edges in a datafarme
    ## whic has index < test_seze, thos should go in test
    # return eids

    test_size = int(len(eids) * test_ratio)
    train_size = g.number_of_edges() - test_size
    print(train_size,test_size)
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    # return train_pos_u, train_pos_v,test_pos_u, test_pos_v

    # Find all negative edges and split them for training and testing
    ## generating aXa matrix which hsows one if edge is present else 0
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),shape=(g.number_of_nodes(),g.number_of_nodes()))

    ## adj.todense to keep only non existent
    ## np.eye to remove identlical edge liek (0,0), (1,1), (2,2)
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    np.random.seed(0)
    ## keeping only half negative samples
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_g = dgl.remove_edges(g, eids[:test_size])
    test_g = dgl.remove_edges(g, eids[test_size:])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    return train_pos_u, train_pos_v,test_pos_u, test_pos_v,train_neg_u, train_neg_v,test_neg_u, test_neg_v,train_g,test_g,train_pos_g,train_neg_g,test_pos_g,test_neg_g


def process_data(image_model,image_included,umap_dt,cols_needed):
    nodes = pd.read_parquet(f'../Dataset/nodes_without_image.parquet')
    edges = pd.read_parquet(f'../Dataset/edges_without_image.parquet',engine='pyarrow')
    # cols_needed = list(nodes.drop(columns=['asin','also_buy']).columns)
    if image_included:
        image_data = pd.read_parquet(f'../Dataset/{image_model}_men_image_features.parquet')
        nodes = pd.merge(nodes,image_data.reset_index().rename(columns={'index':'asin'}),on='asin')
        if umap_dt:
            umap_reducer = UMAP(**umap_dt, random_state=7)
            reduced_image_vectors = umap_reducer.fit_transform(nodes.filter(regex='vec_'))
            nodes = pd.concat([nodes.filter(regex='^(?!vec_)'),pd.DataFrame(reduced_image_vectors,columns=[f'umap_{i}' for i in range(reduced_image_vectors.shape[1])],index=nodes.index)],axis=1)
            cols_needed += list(nodes.filter(regex='umap_').columns)
        else:
            cols_needed += list(nodes.filter(regex='vec_').columns)
    return nodes,edges,cols_needed

def process_data_only_image(image_model,image_included,umap_dt):
    nodes = pd.read_parquet(f'../Dataset/nodes_without_image.parquet')
    edges = pd.read_parquet(f'../Dataset/edges_without_image.parquet',engine='pyarrow')
    if image_included:
        image_data = pd.read_parquet(f'../Dataset/{image_model}_men_image_features.parquet')
        nodes = pd.merge(nodes,image_data.reset_index().rename(columns={'index':'asin'}),on='asin')
        if umap_dt:
            umap_reducer = UMAP(**umap_dt, random_state=7)
            reduced_image_vectors = umap_reducer.fit_transform(nodes.filter(regex='vec_'))
            nodes = pd.concat([nodes.filter(regex='^(?!vec_)'),pd.DataFrame(reduced_image_vectors,columns=[f'umap_{i}' for i in range(reduced_image_vectors.shape[1])],index=nodes.index)],axis=1)
            cols_needed = list(nodes.filter(regex='umap_').columns)
        else:
            cols_needed = list(nodes.filter(regex='vec_').columns)
    return nodes,edges,cols_needed

def save_model(model, optimizer, path):
  """Saves the DGL model, weights, and optimizer to a specified path.

  Args:
      model (dgl.nn.Module): The DGL model to save.
      optimizer (torch.optim.Optimizer): The optimizer used for training (optional).
      path (str): The path to save the model and optimizer state.
  """
  model_state_dict = model.state_dict()
  if optimizer:
    optimizer_state_dict = optimizer.state_dict()
  else:
    optimizer_state_dict = None
  model_data = {
      'model_state': model_state_dict,
      'optimizer_state': optimizer_state_dict
  }
  torch.save(model_data, path)

def get_recommendation_from_graph(graph,model,k):
    edges_g = graph.edges()
    edges_g = pd.DataFrame({'asin':edges_g[0].tolist(),'also_buy':edges_g[1].tolist()})
    true_val_ = edges_g.groupby('asin').also_buy.apply(lambda x: list(x))
    true_val_.index.name = 'node_id'
    scores = get_top_k_new(graph,model,k=k)
    scores_edges = pd.merge(true_val_ , scores, on='node_id')
    return scores_edges

def get_top_k_new(graph, model, k=6):
    """
    Recommends top K products for a given node based on link prediction scores using a trained GraphSAGE model.

    Parameters:
    - graph: The DGL graph.
    - node_id: The ID of the node for which recommendations are to be made.
    - model: The trained GraphSAGE model.
    - k: Number of top products to recommend (default is 5).

    Returns:
    - List of recommended product node IDs.
    """
    
    # Compute node embeddings for the entire graph
    model.eval()
    with torch.no_grad():
        node_embeddings = model(graph, graph.ndata['feat'])
    all_nodes = graph.nodes().cpu().numpy()
  

    recommendations = []
    for i in tqdm(all_nodes):
        link_scores = torch.matmul(node_embeddings[i], node_embeddings.t()).squeeze(0)
        top_k = torch.topk(link_scores, k).indices.cpu().numpy()
        recommendations.append(top_k)
    return pd.DataFrame({"node_id":all_nodes,'recommendations':recommendations})

def create_model_data(image_model,image_included = False,split_ratio =0.3,umap_dt=None,only_image = False,use_bidirectional = False,skip_ndata=False,cols_needed = []):
    # edges = pd.read_parquet(f'../processed/{image_model}/edges.parquet')
    # nodes = pd.read_parquet(f'../processed/{image_model}/nodes.parquet',engine='pyarrow')
    # sub_cat_coded_cols = list(nodes.filter(regex='sub').drop(columns='sub_cat').columns)
    # cols_needed = ['price']+sub_cat_coded_cols
    # if image_included:
    #     cols_needed += list(nodes.filter(regex='vec_').columns)
    # print('node features',len(cols_needed))
    nodes =None
    edges =None
    # cols_needed = None
    if only_image:
        nodes,edges,cols_needed = process_data_only_image(image_model,image_included,umap_dt)
    else:
        nodes,edges,cols_needed = process_data(image_model,image_included,umap_dt,cols_needed)
    if not len(cols_needed):
        print('skipping adding node features')
        temp_cols = nodes.columns
        node_features = nodes[temp_cols].copy()
        for i in temp_cols:
            node_features[i] = 1
        node_features = torch.Tensor(node_features.to_numpy())
    else:
        node_features = torch.Tensor(nodes[cols_needed].to_numpy())
    print("Node data shape :",nodes.shape)
    print("Edges data shape :",edges.shape)
    print("Size of Input features :",len(cols_needed))
    print("node features shape :",node_features.shape)
    # node_labels = torch.from_numpy(text_df['niche'].astype('category').cat.codes.to_numpy())
    edges_src = torch.from_numpy(edges['asin'].to_numpy())
    edges_dst = torch.from_numpy(edges['also_buy'].to_numpy())
    # return nodes,edges
    #Build Graph
    g = dgl.graph((edges_src, edges_dst))
    
    g.ndata['feat'] = node_features
    if use_bidirectional:
        print('##########################\nConverting to biderctional Graph\n##########################')
        g = dgl.to_bidirected(g,copy_ndata=True)
    # g,nodes,edges = generate_graph(df_full[df_full.asin.isin(final_list)],include_img=True)
    train_pos_u, train_pos_v,test_pos_u, test_pos_v,train_neg_u, train_neg_v,test_neg_u, test_neg_v,train_g,test_g,train_pos_g,train_neg_g,test_pos_g,test_neg_g = train_test_split(g,split_ratio)

    return g,nodes,edges,train_pos_u, train_pos_v,test_pos_u, test_pos_v,train_neg_u, train_neg_v,test_neg_u, test_neg_v,train_g,test_g,train_pos_g,test_pos_g,train_neg_g,test_neg_g,cols_needed

def train_model(mlflow,image_model,run_name,n_epochs,model_params,image_included = False,split_ratio =0.3,umap_dt=None,k=100,patience=10,only_image = False,use_bidirectional=False,skip_ndata=False,cols_needed = []):
    
    with mlflow.start_run(run_name=run_name) as run:
        # PATH = './model_checkpoints'
        # if os.path.exists(PATH):
        #     shutil.rmtree(PATH)
        # if not os.path.exists(PATH):
        #     os.makedirs(PATH)
        g,nodes,edges,train_pos_u, train_pos_v,test_pos_u, test_pos_v,train_neg_u, train_neg_v,test_neg_u, test_neg_v,train_g,test_g,train_pos_g,test_pos_g,train_neg_g,test_neg_g,cols_needed = create_model_data(image_model,image_included,split_ratio,umap_dt,only_image=only_image,use_bidirectional=use_bidirectional,skip_ndata=skip_ndata,cols_needed=cols_needed)
        
        # Move the graph to the device
        train_g = train_g.to(device)
        test_g = test_g.to(device)
        train_pos_g = train_pos_g.to(device)
        test_pos_g = test_pos_g.to(device)
        train_neg_g = train_neg_g.to(device)
        test_neg_g = test_neg_g.to(device)
        
        in_feats = train_g.ndata["feat"].shape[1]
        h_feats = model_params['h_feats']
        aggregator_type = model_params['aggregator_type']
        dropout = model_params['dropout']
        lr = model_params['lr']
        ## M
        model = GraphSAGE(in_feats=in_feats,h_feats=h_feats,agg_func=aggregator_type,dropout=dropout).to(device)
        # You can replace DotPredictor with MLPPredictor.
        #pred = MLPPredictor(16)
        pred = DotPredictor().to(device)

        # ----------- 3. set up loss and optimizer -------------- #
        # in this case, loss will in training loop
        torch.manual_seed(0)
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=lr)

        # ----------- 4. training -------------------------------- #
        all_logits = []
        torch.manual_seed(0)
        best_auc = 9999999
        epochs_no_improve = 0
        best_epoch = 0
        for e in range(n_epochs):
            # forward
            model.train()
            h = model(train_g, train_g.ndata['feat'].to(device))
            pos_score = pred(train_pos_g, h)
            neg_score = pred(train_neg_g, h)
            loss = compute_loss(pos_score, neg_score)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if e % 5 == 0:
                print('In epoch {}, loss: {}'.format(e, loss))

            ## Checkpoint mechanism
            
            if loss < best_auc:
                best_auc = loss
                best_epoch = e
                epochs_no_improve = 0
                torch.save(model, f'./best_model.pkl')
                mlflow.log_artifact(f'./best_model.pkl')
                print(f'Epoch {e}: Best model saved min loss {loss}')
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                n_epochs = best_epoch
                print(f'Early stopping at epoch {e} with min loss {best_auc}')
                break
        model = torch.load(f'./best_model.pkl')
        # ----------- 5. check results ------------------------ #
        torch.manual_seed(0)
        with torch.no_grad():
            pos_score = pred(train_pos_g, h)
            neg_score = pred(train_neg_g, h)
            auc_train_score = compute_auc(pos_score, neg_score)
        
        
        torch.manual_seed(0)
        with torch.no_grad():
            model.eval()
            h_test = model(test_g, test_g.ndata['feat'].to(device))
            pos_score = pred(test_pos_g, h_test)
            neg_score = pred(test_neg_g, h_test)
            auc_score = compute_auc(pos_score, neg_score)

        print('Train AUC', auc_train_score)
        print('Test AUC', auc_score)

        scores_test = get_recommendation_from_graph(test_g,model,k)
        scores_train = get_recommendation_from_graph(train_g,model,k)

        recall_test = scores_test.apply(lambda x : len(set(x['also_buy']) & set(x['recommendations']))/len(x['also_buy']) ,axis=1).mean()
        recall_train = scores_train.apply(lambda x : len(set(x['also_buy']) & set(x['recommendations']))/len(x['also_buy']) ,axis=1).mean()

        print(f'Train recall@{k}', recall_train)
        print(f'Test recall@{k}', recall_test)

        mlflow.set_tag("model_name", run_name)
        mlflow.log_param("train_split", split_ratio)
        mlflow.log_param("n_epochs", n_epochs)
        # mlflow.log_param("lr", lr)
        mlflow.log_param("nodes", nodes[cols_needed].shape)
        mlflow.log_param("edges", edges.shape)
        mlflow.log_params(model_params)
        if umap_dt:
            mlflow.log_params(umap_dt)
        torch.save(g, 'graph.pkl')
        mlflow.log_artifact('./graph.pkl')

        torch.save(train_g, 'train_g.pkl')
        mlflow.log_artifact('./train_g.pkl')
        torch.save(train_pos_g, 'train_pos_g.pkl')
        mlflow.log_artifact('./train_pos_g.pkl')
        torch.save(train_neg_g, 'train_neg_g.pkl')
        mlflow.log_artifact('./train_neg_g.pkl')
        
        torch.save(test_g, 'test_g.pkl')
        mlflow.log_artifact('./test_g.pkl')
        torch.save(test_pos_g, 'test_pos_g.pkl')
        mlflow.log_artifact('./test_pos_g.pkl')
        torch.save(test_neg_g, 'test_neg_g.pkl')
        mlflow.log_artifact('./test_neg_g.pkl')

        
        mlflow.log_metrics({f'recall_train_at_{k}':recall_train,f'recall_test_at_{k}':recall_test})
        # mlflow.log_input(nodes[cols_needed].head(), context="training")
        # mlflow.log_artifact("graph", g)
        mlflow.log_param("image_data_included",image_included )
        mlflow.log_metrics({'AUC_train': auc_train_score, 'AUC_test': auc_score})
        # Save model to artifacts
        torch.save(model, 'model.pkl')
        mlflow.log_artifact('./model.pkl')
        
        save_model(model,optimizer,'./model.pt')
        # Save the dictionary to a file using torch.save
        mlflow.log_artifact('./model.pt')
    return model,g,nodes,edges,train_pos_u, train_pos_v,test_pos_u, test_pos_v,train_neg_u, train_neg_v,test_neg_u, test_neg_v,train_g,test_g
    