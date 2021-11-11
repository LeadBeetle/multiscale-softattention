from collections import Counter
from itertools import count
from networkx.classes.function import degree, degree_histogram
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset
from utils.constants import * 
from utils.utils import * 
from torch_geometric.datasets import Planetoid


class Graphsave:
    
    def __init__(self, dataset):
        root = osp.abspath( 'data')
        self.dataset_name = dataset
        isPlanetoid = self.dataset_name in [Dataset.CORA, Dataset.PUBMED, Dataset.CITESEER]
        
        if isPlanetoid:
            self.dataset = Planetoid(root, name=datasetMapping[self.dataset_name])        
        else:
            self.dataset = PygNodePropPredDataset(datasetMapping[self.dataset_name], root)
        self.device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
        self.data = self.dataset[0]
        self.num_nodes = self.data.num_nodes
        self.num_classes = self.dataset.num_classes
        self.num_features = self.dataset.num_features
        self.x = self.data.x.to(self.device)
        self.y = self.data.y.squeeze().to(self.device)
        

    def createGraph(self):

        edge_index = self.data.edge_index
        #Create Graph

        strtNode = edge_index[0][edge_index.size()[1]-1]
        ndNode = edge_index[1][edge_index.size()[1]-1]
        hghstNode=strtNode.item()
        Graph = nx.Graph()
        for i in range(hghstNode) :
            clss=self.y[i]
            clss =clss.item()            
            Graph.add_node(i, cat =clss)
        for j in range(edge_index.size()[1]-1):
            frstNode = edge_index[0][j]
            scndNode = edge_index[1][j]
            Graph.add_edge(frstNode.item(),scndNode.item())
        Graph.remove_node(hghstNode)
        print(degree_histogram(Graph))
        return Graph
         
    def plotGraph(self, Graph):
        groups = set(nx.get_node_attributes(Graph,'cat').values())
        mapping = dict(zip(sorted(groups),count()))
        d=dict(Graph.degree)
        
        colors = [mapping[Graph.nodes[n]['cat']] for n in range(len(Graph.nodes()))]
        
        pos = nx.kamada_kawai_layout(Graph,scale=1000)
        ec = nx.draw_networkx_edges(Graph, pos, width=0.01)
        nc = nx.draw_networkx_nodes(Graph, pos, nodelist=Graph.nodes(), node_color=colors, node_size=1 , cmap=plt.cm.jet)
        #node_size=[v*5 for v in d.values()]
        plt.colorbar(nc)
        plt.axis('off')
        plt.show()
        
    def plotHistogramm(self, Graph):
        print(degree_histogram(Graph))    
        degrees = [Graph.degree(n) for n in Graph.nodes()]
        plt.hist(degrees,bins=len(degree_histogram(Graph)))
        plt.show()
        

    def getPathesUnderLength(self, Graph, length):

        path =dict(nx.all_pairs_shortest_path(Graph,length))
        valueList =[]
        for ite in range(len(path)):
            for keys in path[ite]:
                if len(path[ite][keys])>1:
                    edgeStrt=path[ite][keys][0]
                    edgeNd=path[ite][keys][len(path[ite][keys])-1]
                    tuple=(edgeStrt,edgeNd)
                    valueList.append(tuple)
            #alte Histogrammfunktion; vllt noch usefull
               
            #valueList.append(len(path[ite])-1)    
        #cnt=Counter(valueList).items()
        #cnt=sorted(cnt) 
        #x,y=zip(*cnt)
        #plt.bar(x,y)
        #plt.show()
        return valueList
        
        
        
    def getPathsOfCertainLengths(self,Graph, length):
        
        path =dict(nx.all_pairs_shortest_path(Graph,length))
        valueList =[]
        for ite in range(len(path)):
            for keys in path[ite]:
                if len(path[ite][keys])==length:
                    edgeStrt=path[ite][keys][0]
                    edgeNd=path[ite][keys][length-1]
                    tuple=(edgeStrt,edgeNd)
                    valueList.append(tuple)

        return valueList
    
    def addEdgesToGraph(self,Graph, edges):
        Graph.add_edges_from(edges)
        return Graph 