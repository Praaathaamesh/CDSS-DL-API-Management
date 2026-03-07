# Instructions to model training checkpoint 1
### *Author* : *Prathamesh Pradeep Jadhav*

## Component Pipelines
- *CheckPointCase.ipynb* : First trail pipeline using 1DCNN + Bi-LSTM; no residual blocks; thresholding to metrics like PR-AUC, recall, precision for imporved performance. Requires optimisation and refactoring of functions.
- *PROS_Z_Checkpoint.ipynb* : First pipeline to include Partial Random Oversampling w/o Replacement (PROS) with Z-score normalisation of signals. Code blocks arranged and some optimisations were done.
- *STFT_Checkpoint.ipynb* : Tested Short Term Fourier Transform (STFT) signal normalisation technique to use 2DCNN instead of 1DCNN.
- *Z-Checkpoint.ipynb* : Only signal Z-score normalisation done.

## Key Considerations
- All the splits used were previously made and pickled using data preprocessing pipeline--including all the metadata and signal preprocessing.
- Hamming Loss/Score were subclassed from Keras API--since there were some issues installing the TensorFlow Addons package in the local system. Most likely Newer TensorFlow and Python3 version clashes.
- PROS introduced heavy overfitting despite oversampling the less frequent entries--specifically high risk entries. No algorithm was created as per the research paper referred. Hence, employed it myself by replacing the randomising notion with systematic duplication of important low-frequency entries.

## Performance notes
- All the checkpoint pipelines has failry similar and suboptimal results, when compared the precise nature of model's predictive capabilities (lower precision, ROC-AUC, PR-AUC and moderate F2 score).
- Despite the results, some data processing issues are suspected. That is why, next performance checkpoint won't include any signal preprocessing. Metadata preprocessing will persist.