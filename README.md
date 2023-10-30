# Optimal Surrogate Models for Predicting Elastic Moduli of Metal-Organic Frameworks via Multiscale Features 
This repository introduces the multiscale features for predicting the mechanical properties of MOFs.

# Abstract
Evaluating the mechanical stability of metal-organic frameworks (MOFs) is essential for their successful application in various fields. Therefore, the objective of this study was to develop optimal machine learning (ML) models for predicting the bulk and shear moduli of MOFs. Considering the effects of global (such as porosity and topology) and local features (including metal nodes and organic linkers) on the mechanical stability of MOFs, we developed multiscale features that can incorporate both types of features. To this end, we first explored descriptors representing the global and local features of MOFs from datasets of previous studies in which elastic moduli were computed. We then assessed the performance of various combinations of these descriptors to determine the optimal multiscale features for predicting the elastic moduli. The optimal surrogate models trained using multiscale features exhibited R2 values of 0.868 and 0.824 for bulk and shear moduli, respectively. Furthermore, the surrogate models outperformed prior benchmarks. Finally, through model interpretation, we discovered that for similar pore sizes, metal nodes are the most dominant factor affecting the mechanical properties of MOFs. We anticipate that our approach will be a valuable tool for future research on the discovery of mechanically robust MOFs for various industrial applications.

<img width="80%" src="https://ifh.cc/g/4FQKrA.png](https://ifh.cc/g/HcrFXP.jpg"/>

# SurrogateModel_MOF_Mechanical_Stability

Data and scripts for predicting the mechanical properties of MOFs

## Install

Note: This packages is tested on the Linux and mac. We recommended using Linux or Mac for implementing scripts.

## Dependencies
This project currently requires the following packages:

* pandas 1.5.1
* matplotlib 3.5.2
* numpy 1.20.3
* sklearn
* pycaret 2.3.10
  
## Download the datasets

- You can download the datasets in [my figshare link](https://doi.org/10.6084/m9.figshare.24316339).
- All the CIF files and features that we utilized in our study can be found in the above link.
- We recommend not changing the configuration of the folders.

## Scripts

* OptimalSurrogateKVRH.ipynb and OptimalSurrogateGVRH.ipynb contains the results of benchmarking.

* Trainingsize.ipynb files contains the results of the robustness of each descriptor in terms of training size.
