# Instructions to model training checkpoint 2
### *Author* : *Prathamesh Pradeep Jadhav*

## Component Pipelines
- *Z-Checkpoint.ipynb* : Trail pipeline using 1DCNN + Bi-LSTM; no residual blocks; thresholding to metrics like PR-AUC, recall, precision for imporved performance. Requires slight optimisation and refactoring of functions. Z-score normalisation of signal data.

## Key Considerations
- All the splits used were previously made and pickled using data preprocessing pipeline-- not including any of the signal preprocessing. Metadata preprocessing persists.
- Hamming Loss/Score were subclassed from Keras API--since there were some issues installing the TensorFlow Addons package in the local system. Most likely Newer TensorFlow and Python3 version clashes.

## Performance notes
- All the checkpoint pipelines have failry similar and suboptimal results, when compared the precise nature of model's predictive capabilities (lower precision, ROC-AUC, PR-AUC and moderate F2 score).
- Similar performance bottlenecks as per previous checkpoints.
- Despite the results, some data processing issues are still suspected. That is why, next performance checkpoint won't include any preprocessing and pickling. Data will be directly loaded from original database directory.