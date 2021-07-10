# Graph Neural Networks with Multiscale Soft Attention

## Table Of Contents
- [Einleitung](#einleitung)
- [Aufgabenstellung](#aufgabenstellung)
- [Code Explainer](#code-explainer)

## Einleitung

Dieses Repository umfasst den Code und die Dokumente, die bei der Untersuchung von GATs und Graph Transformern hinsichtlich ihrer Erweiterung um eine Multi-Scale-Soft-Attention im Rahmen eines Projektseminars an der TU Ilmenau entstanden sind. 

Die erste Themenbeschreibung befindet sich hier: [Themenbeschreibung](Dokumente/Multi-Scale-SoftAttention-Topic.pdf)  
Bisher von uns eingesetzte Literatur liegt in: [Literatur](Dokumente/Literatur)

Bisher von Marco bereitgestellte Literatur: 

- [Graph attention networks](https://arxiv.org/abs/1710.10903)
- [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067)
- [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198)
- [A generalization of transformer networks to graphs](https://arxiv.org/abs/2012.09699)
- [Fast Graph Representation Learning with PyTorch Geometric](https://arxiv.org/abs/1903.02428)
- [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491)
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478)


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

| GAT-V1-Datasets | Nodes | Edges | Classes | 
| :-------------: | :-------------: | :-------------: | :-------------: | 
| [Cora](https://paperswithcode.com/dataset/cora) | 2.708  |  5.429 | 7 | 
| [Citeseer](https://paperswithcode.com/dataset/citeseer) |  3.312  | 4.732  | 6 | 
| [Pubmed](https://paperswithcode.com/dataset/pubmed)|  19.717 |  44.338  | 3 | 
| [PIP](https://paperswithcode.com/dataset/ppi) |  Ø 2.373 |  -  | 121 | 

| GAT-V2-Datasets | Nodes | Edges | Classes |
| :-------------: | :-------------: | :-------------: | :-------------: | 
| [ogbn-arxiv](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-arxiv) | 169.343 |  1.166.243 | 40 | 
| [ogbn-products](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-products) | 2.449.029 |  61.859.140 | 47 | 
| [ogbn-mag](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-mag) | 1.939.743 |  21.111.007 | 349 | 
| [ogbn-proteins](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-proteins) | 132.534 |  39.561.252 | 8 | 

