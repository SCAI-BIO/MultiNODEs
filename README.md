# MultiNODEs

## Description

This repository contains a new organized version of the <b>MultiNODEs:</b> project for running the model using the PPMI dataset.

Full details of the approach are on the [paper](https://www.nature.com/articles/s41746-022-00666-x). 

## Data

To execute the code you need to download PPMI dataset, save the files in a folder data/PPMI, and organize them according to what MultiNODE's paper explains.


## Download

To download everything from this repository onto your local directory, execute the following line on your terminal:
```
$ git clone https://gitlab.scai.fraunhofer.de/diego.felipe.valderrama.nino/multinodes_newpublicversion.git MultiNODEs
$ cd MultiNODEs
```

## Run

First you need to create an anaconda environment. Please use the following command:
```
$ conda env create -f requirements.yml
$ conda activate MultiNODEs
```

Update the paths in the parser.py to the ones of your system

To train the code with the best hyparameters for PPMI dataset and generate simualations using reconstruction, prior and posterior sampling please use
```
$ python main.py --exp_name=MultiNODEs_PPMI --static_data=True --GPU=0 --mode=train_full
```

If you just want to train the model without generate simulations please use
```
$ python main.py --exp_name=MultiNODEs_PPMI --static_data=True --GPU=0 --mode=only_train
```

If you just want to generate simulations using the reconstruction configuration. Please change the epoch_init parameter to the one you need:
```
$ python main.py --exp_name=MultiNODEs_PPMI --static_data=True --GPU=0 --epoch_init=1900 --mode=only_rec
```

If you just want to generate simulations using the prior sampling . Please change the epoch_init parameter to the one you need:
```
$ python main.py --exp_name=MultiNODEs_PPMI --static_data=True --GPU=0 --epoch_init=1900 --mode=only_prior
```

If you just want to generate simulations using the posterior sampling. Please change the epoch_init parameter to the one you need:
```
$ python main.py --exp_name=MultiNODEs_PPMI --static_data=True --GPU=0 --epoch_init=1900 --mode=only_posterior
```

You can train the model using different parameters just writing them when running the model or modifying their default values. Please see the parser.py script


## Visualization

When you want to generate some visualizations, you need a trained model. Then you can use the following command
```
python visualizations.py --exp_name=MultiNODEs_PPMI --static_data=True --num_epoch=1900
```

Please be sure that --num_epoch is the number of epoch of the model you want to load and that --sigma_long and --sigma_stat are the same as during training

# Repo organization


    ├── LICENSE
    ├── README.md                  
    ├── data
    │   ├── PPMI                   <- PPMI dataset.
    │       └── ... .csv           <- csv files
    │
    ├── models                     <- Trained models. The code generate all the folders as follows:
    │   └── Exp Name               <- Folder with the experiment name.
    │       ├──train_opt.txt       <- Train options
    │       ├──losses              <- Folder with information of losses (graph, txt and excel sheet)
    │       ├──models              <- Folder with the checkpoints
    │       └──samples             <- Folder with reconstruction, prior and posterior generation dynamics
    │           └──Epxxxx          <- Folder with the images you generate from visualization.py using Ep xxxx model
    │
    ├── requirements.yml           <- The requirements file for reproducing the analysis environment
    │                                  
    │
    ├── src                        <- Source code for use in this project.
    │   ├── __init__.py            
    │   │
    │   ├── data                   <- Scripts to load the datasets
    │   │   ├── __init__.py        
    │   │   └── datasets.py        <- Script to create each dataset to be used in the dataloader
    │   │   └── load_data.py       <- Script to load the data and datasets according to the parser
    │   │
    │   ├── models                 <- Main scripts to generate trajectories using reconstruction, prior or
    │   │   │                         posterior sampling
    │   │   ├── __init__.py        <- Makes models a Python module
    │   │   └── ....py
    │   │
    │   └── visualization          <- Scripts to create exploratory and results oriented visualizations
    │       ├── __init__.py        <- Makes visualization a Python module
    │       └── visualizations.py  <- Script to plot some visualizations
    │



--------
## Contact
- [Prof. Dr. Holger Fröhlich](mailto:holger.froehlich@scai.fraunhofer.de)
- [Diego Valderrama](mailto:diego.felipe.valderrama.nino@scai.fraunhofer.de)
- AI and Data Science Group, Bioinformatics Department, Fraunhofer Institute for Algorithms and Scientific Computing (SCAI), Schloss Birlinghoven, 1, 53757 Sankt Augustin.
