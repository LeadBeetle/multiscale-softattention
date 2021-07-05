# Graph Neural Networks with Multiscale Soft Attention

Dieses Repository umfasst den Code und die Dokumente, die bei der Untersuchung von GATs und Graph Transformern hinsichtlich ihrer Erweiterung um eine Multi-Scale-Soft-Attention im Rahmen eines Projektseminars an der TU Ilmenau entstanden sind. 

Die erste Themenbeschreibung befindet sich hier: [Themenbeschreibung](../Dokumente/Multi-Scale-SoftAttention-Topic.pdf) 
Bisher von uns eingesetzte Literatur liegt in: [Literatur](../Dokumente/Literatur)

Bisher von Marco bereitgestellte Literatur: 

- [Graph attention networks](https://arxiv.org/abs/1710.10903)
- [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067)
- [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198)
- [A generalization of transformer networks to graphs](https://arxiv.org/abs/2012.09699)
- [Fast Graph Representation Learning with PyTorch Geometric](https://arxiv.org/abs/1903.02428)


**Aufgabenstellung:**

1. Literaturrecherche
2. Einbeziehung höherer Nachbarschaftsordnung in GATs
3. Einbeziehung höherer Nachbarschaftsordnung in Graph Transformern
4. Node-Classification-Experimente auf Daten
    1.  Untersuchung der Attentiongewichte - inwiefern werden Nachbarn höherer Ordnung einbezogen?
    2.  Hat separierte Berechnung durch K Äste Vorteile gegenüber der einmaligen Berechnung
    3.  Wie korreliert der maximale Grad der Nachbarschaft mit der potenziellen Verringerung der Netzwerktiefe
5. Auswertung der Ergebnisse
