#!/bin/bash

# 0 for false 1 for true
ShowingModel=0
ModelName=efficientb0
DatasetPath=datasets/ts_data
Epochs=1
ReSize=256
BatchSize=16
Optimizer=adam
Scheduler=StepLR
LrRate=0.0002

python main.py --ShowingModel $ShowingModel --ModelName $ModelName --DatasetPath "$DatasetPath" --Epochs $Epochs --ReSize $ReSize --BatchSize $BatchSize --Optimizer $Optimizer --Scheduler $Scheduler --LrRate $LrRate