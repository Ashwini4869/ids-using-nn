# Project Logs  
Just a simple logs of my work done on this project. You can safely ignore this!

### Date: 2025-06-02
- Implemented training pipeline
- Sample run the pipeline -> overfitting observed

### Date: 2025-06-12
- Add file logging feature
- Model is overfitting on the dataset (validation loss increasing while training loss is decreasing) -> Need to try out regularization techniques(add weight decay, dropout, decrease batch size)
- Adding weight decay leads to a more stable training
- Need to add inference and testing pipeline for final accuracy and F1-Score

### Date: 2025-06-13
- Decreasing Batch size to 64 leads to an even stable training
- Decreasing Batch size to 32 leads to unstable training

### Date: 2025-06-18
- Adding an extra layer to the model does not perform well
- Add testing pipeline 
- Save model after training
- Add multiple layers with dropout -> improves performance
- Increasing weight decay to 0.02 and lr to 0.01 -> does not converge at all
- Update training hyperparameters and model architecture
- Migration to 17 server for faster training

### Date: 2025-06-19
- Reported F1-Score: 0.7
- Try different overfitting techniques and hyperparameters
- Sample train SVM and random forest on the dataset for comparison
- SVM and RF are able to perform almost at the same level of NN

### Date: 2025-06-20
- Modified the model and hyperparameters to get F1-Score of 0.7221 (model name: model_2025-06-20-15-54_best.pt )
- Need to use additional measures for overfitting and already trained Neural Networks on NSL-KDD