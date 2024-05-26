@echo off
REM 0 for false 1 for true
SET ShowingModel=0
SET ModelName=mix
SET DatasetPath=datasets/merged_split
SET Epochs=30
SET ReSize=256
SET BatchSize=32
SET Optimizer=Adam
SET Scheduler=StepLR
SET LrRate=0.001

python main.py --ShowingModel %ShowingModel% --ModelName %ModelName% --DatasetPath "%DatasetPath%" --Epochs %Epochs% --ReSize %ReSize% --BatchSize %BatchSize% --Optimizer %Optimizer% --Scheduler %Scheduler% --LrRate %LrRate%