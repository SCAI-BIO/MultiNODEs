import os

import optuna

optuna.create_study(study_name=os.environ['STUDY'], storage=os.environ['DBURL'], sampler=optuna.samplers.TPESampler(), direction='minimize')
    

