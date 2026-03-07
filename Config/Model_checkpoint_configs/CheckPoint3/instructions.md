# Instructions to model training checkpoint 3
### *Author* : *Prathamesh Pradeep Jadhav*

## Component Pipelines
- *1DCNN-BiLSTM_Resnet_Direct.ipynb* : Trail pipeline using *1DCNN + Bi-LSTM with residual blocks*; thresholding to precision for an imporved performance and to fit the problem statement gracefully *(Precision-recall tradeoff marked with preference to higher recall)*. Proper optimisation and function refactoring is done. Z-score normalisation *per lead* of signal data. Ditched the PROS.

## Key Considerations
- All the splits used were *created and preprocessed in the pipeline* itself--_not_ including any of the signal preprocessing and metadata preprocessing from Data processing pipeline.
- No use of pickled variables.
- Hamming Loss were subclassed from Keras API--since there were some issues installing the TensorFlow Addons package in the local system. Most likely Newer TensorFlow and Python3 version clashes.
- Later, for increased training time, Bi-LSTM blocks were nullled (No significant effect on the performance).

## Performance notes 
- When compared to the previous performance pipelines, the metrics have improved *significantly*--fitting the targets fairly well. 
- NO performance bottlenecks were seen as per previous checkpoints.
- Despite the results, some data processing is required to label the high risk entries. That is why, next performance checkpoint will include HYP to HYP_HR relabling module with the reference of preprocessing pipeline. Data will be directly loaded from original database directory.