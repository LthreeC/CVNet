@echo off
REM 0 for false 1 for true
SET ShowingModel=0
SET ModelName=efficientb0
SET DatasetPath=datasets/ts_data
SET Epochs=1
SET ReSize=256
SET BatchSize=16
SET Optimizer=adam
SET Scheduler=StepLR
SET LrRate=0.0002

python main.py --ShowingModel %ShowingModel% --ModelName %ModelName% --DatasetPath "%DatasetPath%" --Epochs %Epochs% --ReSize %ReSize% --BatchSize %BatchSize% --Optimizer %Optimizer% --Scheduler %Scheduler% --LrRate %LrRate%