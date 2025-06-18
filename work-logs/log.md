# Project Logs

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

