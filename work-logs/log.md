# Project Logs

### Date: 2025-06-02
- Implemented training pipeline
- Sample run the pipeline -> overfitting observed

### Date: 2025-06-12
- Add file logging feature
- Model is overfitting on the dataset (validation loss increasing while training loss is decreasing) -> Need to try out regularization techniques(add weight decay, dropout, decrease batch size)
- Adding weight decay leads to a more stable training
- Need to add testing pipeline for final accuracy and F1-Score
