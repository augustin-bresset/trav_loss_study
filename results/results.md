# Resultats

Je vais écrire ici les résultats de mes analyses.


## Metriques

* IoU : Intersection over Union
* F1
* F2 
...


## Loss
### BCE Binary Cross Entropy (POS WEIGHT)

We use BCE with a weight 

### Focal
Hyperparameter : 
* Position Weight
* Gamma 

If gamma = 0 : Focal <=> BCE

### Asymetric
Hyperparameter : 
* Position Weight
* Gamma positive
* Gamma negative (equivalent gamma focal)

If gamma positive = 0 : Asymetric <=> Focal




## Model 



| Expérience | F1 | F2 | IoU | Precision | Recall | FPR collision | PR-AUC | ECE |
  |---|---|---|---|---|---|---|---|---|
  | fold0 / minkunet_cr025 | 0.1039 | 0.0676 | 0.0548 | 0.967 | 0.055 | 0.0005 | 0.639 | 0.197 |
  | fold0 / minkunet_cr05  | 0.1141 | 0.0746 | 0.0605 | 0.966 | 0.061 | 0.0006 | 0.657 | 0.196 |
  | fold0 / minkunet_cr10  | NaN    | —      | —      | —     | —     | —      | —     | —     |
  | fold0 / travnet_cr025  | 0.1168 | 0.0764 | 0.0620 | 0.965 | 0.062 | 0.0006 | 0.391 | 0.171 |
  | fold0 / travnet_cr05   | 0.1184 | 0.0775 | 0.0629 | 0.972 | 0.063 | 0.0005 | 0.653 | 0.196 |
  | fold0 / travnet_cr10   | 0.1101 | 0.0719 | 0.0583 | 0.977 | 0.058 | 0.0004 | 0.647 | 0.197 |
  | fold1 / travnet_cr025  | **0.1561** | **0.1040** | **0.0846** | 0.942 | 0.085 | 0.0015 | 0.617 | 0.188 |