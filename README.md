# Graph Neural Networks with Multiscale Soft Attention

## Table Of Contents
- [Graph Neural Networks with Multiscale Soft Attention](#graph-neural-networks-with-multiscale-soft-attention)
  - [Table Of Contents](#table-of-contents)
  - [Einleitung](#einleitung)
  - [Aufgabenstellung](#aufgabenstellung)
  - [Code Explainer](#code-explainer)
  - [Datensätze](#datensätze)

## Einleitung

Dieses Repository umfasst den Code und die Dokumente, die bei der Untersuchung von GATs und Graph Transformern hinsichtlich ihrer Erweiterung um eine Multi-Scale-Soft-Attention im Rahmen eines Projektseminars an der TU Ilmenau entstanden sind. 

Die erste Themenbeschreibung befindet sich hier: [Themenbeschreibung](Dokumente/Multi-Scale-SoftAttention-Topic.pdf)  
Bisher von uns eingesetzte Literatur liegt in: [Literatur](Dokumente/Literatur)

Bisher von Marco bereitgestellte Literatur: 

- [Graph attention networks (GATV1)](https://arxiv.org/abs/1710.10903)
- [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067)
- [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198)
- [A generalization of transformer networks to graphs](https://arxiv.org/abs/2012.09699)
- [Fast Graph Representation Learning with PyTorch Geometric](https://arxiv.org/abs/1903.02428)
- [How Attentive are Graph Attention Networks? (GATV2)](https://arxiv.org/abs/2105.14491)
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478)
- [Label Prediction With Graph Transformers](https://arxiv.org/abs/2009.03509)


## Aufgabenstellung

1. Literaturrecherche
2. Einbeziehung höherer Nachbarschaftsordnung in GATs
3. Einbeziehung höherer Nachbarschaftsordnung in Graph Transformern
4. Node-Classification-Experimente auf Daten
    1.  Untersuchung der Attentiongewichte - inwiefern werden Nachbarn höherer Ordnung einbezogen?
    2.  Hat separierte Berechnung durch K Äste Vorteile gegenüber der einmaligen Berechnung
    3.  Wie korreliert der maximale Grad der Nachbarschaft mit der potenziellen Verringerung der Netzwerktiefe
5. Auswertung der Ergebnisse

## Code Explainer

Ein gutes Video kann hier gefunden werden: [Code Explainer](https://www.youtube.com/watch?v=364hpoRB4PQ)  
Zu PyTorch-Geometric gibt es eine Videoreihe : [PyTorch-Geometric](https://www.youtube.com/playlist?list=PLGMXrbDNfqTzqxB1IGgimuhtfAhGd8lHF)  

## Datensätze 

In den Papern zu den beiden GAT-Version wurden Benchmarks für die folgenden Datensätze eingefahren.  

In der folgenden Tabelle ist zu erkennen, dass in [Graph attention networks](https://arxiv.org/abs/1710.10903) sowohl im **induktiven** als auch im **transduktiven** Setting experimentiert und dabei über die **Accuracy** ausgewertet wurde.

| GAT-V1-Datasets | Nodes | Edges | Classes | Transductive | Inductive |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| [Cora](https://paperswithcode.com/dataset/cora) | 2.708  |  5.429 | 7 | 83.0 | - |
| [Citeseer](https://paperswithcode.com/dataset/citeseer) |  3.312  | 4.732  | 6 | 72.5 | - |
| [Pubmed](https://paperswithcode.com/dataset/pubmed)|  19.717 |  44.338  | 3 | 79.0 | - |
| [PIP](https://paperswithcode.com/dataset/ppi) |  56.944 |  818.716  | 121 | - | 0.973 |

Im Gegensatz dazu wurde in [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) ausschließlich im induktiven Setting experimentiert. Im folgenden sind GAT-V1 und GAT-V2 auf den OGBN-Daten gegenübergestellt. Auf den obigen Daten wurde im Rahmen des Papers nicht experimentiert.  

| GAT-V2-Datasets | Nodes | Edges | Classes | GAT-V1 | GAT-V2 | Transformer |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------:|
| [ogbn-arxiv](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-arxiv) | 169.343 |  1.166.243 | 40 | 71.54(1h) | 71.87(1h) | **73.11** |
| [ogbn-products](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-products) | 2.449.029 |  61.859.140 | 47 | 77.23(1h) | 80.63(1h) | **82.56** |
| [ogbn-mag](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-mag) | 1.939.743 |  21.111.007 | 349 | 32.20(1h) | **32.61**(1h) | - |
| [ogbn-proteins](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-proteins) | 132.534 |  39.561.252 | 8 | 78.63(8h) | **79.52**(8h) | **86.42** |
