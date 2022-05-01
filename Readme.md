# Multimodal Neural Ordinary Differential Equations

This is the implementation of the Multimodal Neural Ordinary Differential Equations (MultiNODEs) presented in Wendland, Birkenbihl etal. [1]. The MultiNODE approach is a generative model which extend the Neural ODEs [2] to deal with missing data and static variables. As a consequence of that the MultiNODEs can handle and generate new patient trajectories. 
The Multimodal Neural ODE code adapts some code of other repositories. Therefore, please cite in addition to the MultiNODE paper [1] also the Neural Ordinary Differential Equations [2] paper, the Handling Incomplete Data using VAEs paper [3] and the Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE paper [4], if you use this code for your own research.
Our code is build on top of the Neural ODE code from [2] and extend their code. Therefore some parts of the code of [3] and [4] were used.

Explanations of the computed methods can be find in the main.py file and in [1].

**Important**: The data used in our analysis is not included in this repository - please contact PPMI (https://www.ppmi-info.org/access-data-specimens/download-data/) and NACC (https://naccdata.org/requesting-data/submit-data-request) to get access to the data, if you want to run our code with the data used in [1].

# Content of the repository
* Plots of the paper (and supplementary plots of all variables) for the PPMI data and the NACC data in the folder "Plots"
* The main skripts to run the analysis of the PPMI data in the "code" folder 
* The analysis of the SIR data in the "SIR-example" folder

# How to run the Code

* Main skripts
 * For the hyperparameteroptimization it is necessary to run the hyperparameteroptimization_create_study.py file and the hyperparameteroptimization.py file (these files need to be started with a bash skript on a cluster - or have to be changed to run locally). 
 * To train the model (and synthetisize virtual patients) you have to run the training.py file.
* Skripts to analyze SIR
 * The file to generate the SIR data is called SIR-model.py.
 * Similar to the main skripts for the hyperparameteroptimization it is necessary to run the hyperparameteroptimization_create_study.py file and the hyperparameteroptimization.py files. 
 * To train the model you have to run the train files.
 * In addition to the training files we upload some plot skripts to generate the plots
 * The bnet.R file generates via a bayesian network virtual patients. To use this file it is necessary to run in the training file the reconrp function with save_latent=True to save the latent representation of the data. Afterwards it is possible to run the generationprior function of the training.py script

# List of Package dependencies
To run our code it is necessary to install the following packages

* numpy [5]
* pytorch [6]
* pandas [7]
* matplotlib [8]
* scipy [9]
* sklearn [10]
* optuna [11]

*Note*: If you want to use the torch_ACA ODE solver, you have to download the code manually and change the directory in the code in the variable "solver" (like in the . You can find the code of the torch_ACA folder here (https://github.com/juntang-zhuang/torch_ACA/tree/dense_state2/torch_ACA). In our implementation we use their code before it was integrated in the TorchDiffEqPack. We never tested the TorchDiffEqPack code of the package, but you can use it by using solver == 'torchdiffeqpack'. When using the TorchDiffEqPack, please cite [4] and [12]

# References 
[1] Philipp Wendland, Colin Birkenbihl, Marc Gomez-Freixa, Maik Kschischo, Holger Fröhlich. "Generation of realistic synthetic data using multimodal neural differential equations". 2021

[2] Ricky T. Q. Chen, Yulia Rubanova etal. "Neural Ordinary Differential Equations." Advances in Neural Information Processing Systems. arXiv: 1806.07366. 2018

[3] Alfredo Nazabal, Pablo M. Olmos etal. "Handling Incomplete Data using VAEs". arXiv:1807.03653. 2020

[4] Juntang Zhuang, Nicha Dvornek etal. "Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE". Proceedings of the 37th International Conference on Machine Learning, PMLR 119:11639-11649. 2020

[5] Stéfan van der Walt, S Chris Colbert etal. "The NumPy Array: A Structure for Efficient Numerical Computation". Computing in Science & Engineering 13.2, p. 22–30. ISSN : 1521-9615. DOI : 10.1109/MCSE.2011.37

[6] Adam Paszke, Sam Gross etal. "PyTorch: An Imperative Style, High-Performance Deep Learning Library". Advances in Neural Information Processing Systems 32, p. 8024-8035. 2019

[7] Wes McKinney. "Data Structures for Statistical Computing in Python". Python in Science Conference. Austin, Texas, 2010, S. 56–61. DOI : 10.25080/Majora-92bf1922-00a

[8] John D. Hunter. "Matplotlib: A 2D Graphics Environment". Computing in Science & Engineering 9.3 (2007), S. 90–95. ISSN : 1521-9615. DOI : 10.1109/MCSE.2007.55

[9] Paul Virtanen, Ralf Gommers etal. "SciPy 1.0: fundamental algorithms for scientific computing in Python". Nature Methods 17.3, p. 261–272. ISSN :1548-7091, 1548-7105. DOI : 10.1038/s41592-019-0686-2 

[10] Fabian Pedregosa, Gaël Varoquaux etal. "Scikit-learn: Machine Learning in Python". arXiv:1201.0490 2018

[11] Takuya Akiba, Shotaro Sano etal. "Optuna: A Next-generation Hyperparameter Optimization Framework“. arXiv:1907.10902. 2019

[12] Juntang Zhuang, Nicha Dvornek etal. "MALI: A memory efficient and reverse accurate integrator for Neural ODE" International Conference on Learning Representations

**Contact**: Philipp Wendland - wendland.philipp@web.de
