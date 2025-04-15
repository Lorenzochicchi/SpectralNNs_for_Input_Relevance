# Spectral Neural Networks for automatic input feature relevance estimation

In this repository is stored the code used in the paper "Automatic Input Feature Relevance via Spectral Neural Networks".

There are four distinct notebooks, one for each dataset considered in this work. Each notebook contains the code for model definition, dataset generation, model training, and result plotting.

In the case of the stellar spectra dataset, a reduced version of the dataset used in the study is provided to test the method.

The method consists of adding a Spectral Layer as first operation of the model. The Spectral layer is essentialy a Dense Layer with weigths that are parametrized as $\w_{ij} = \lmbda_i \phi_{ij}$. 

![image](https://github.com/user-attachments/assets/f7e1993d-aa64-4f03-9bd8-cee1710cf801)

