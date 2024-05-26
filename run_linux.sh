#!/bin/bash

# 0 for false 1 for true
ShowingModel=0
ModelName=medmamba
DatasetPath=datasets/merged_split_sampled
Epochs=30
ReSize=256
BatchSize=32
Optimizer=Adam
Scheduler=StepLR
LrRate=0.005

python main.py --ShowingModel $ShowingModel --ModelName $ModelName --DatasetPath "$DatasetPath" --Epochs $Epochs --ReSize $ReSize --BatchSize $BatchSize --Optimizer $Optimizer --Scheduler $Scheduler --LrRate $LrRate