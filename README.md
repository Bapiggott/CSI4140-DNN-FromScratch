
# Neural Network Implementation  

## Description  
This project implements custom neural network components, including:  
- A fully connected neural network (FCNN)  
- Convolutional neural network (CNN)  

## Project Structure  
- **Main Components**:  
  - `custom_dropout(x, p)`: Implements dropout regularization  
  - `ExponentialLearningRateDecay`: Implements learning rate decay  
  - `AdamOptimizer`: Custom Adam optimizer implementation  
  - `FullyConnectedNN`: Custom fully connected neural network  
  - `ConvolutionNN`: Custom convolutional network  
  - `MaxPool`: Implements max pooling  

- **Model Building & Training**:  
  - `build()`: Initializes the convolutional and fully connected layers  
  - `to(device)`: Moves model parameters to CPU or GPU  
  - `forward(x)`: Performs a forward pass  
  - `backward(grad_out)`: Computes gradients for backpropagation  

## Requirements  
install dependencies to run (Note if not wanting to use wandb, simply comment out such lines):  
```bash
pip install torch torchvision tqdm wandb
```

## How to Run  
1. **Unzip dirctory**  

2. **Run the script**  
Execute the Python script using (may be different depending on envirnment):  
```bash
python3 main_file.py
```

