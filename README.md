# Write2Predict Overall
A web demo for real-time handwriting recognition using Pytorch and Streamlit.
Users can draw a digit or a letter on the canvas, then the AI model will predict 
the character.

# Dataset
The dataset used is the EMNIST Balanced dataset which contains the image samples of digitals and letters
but merge some confused letters (e.g. `C` and `c`) to 47 classes. All images are in a 28x28 pixel grayscale format.

# Train the Model
All codes and results are in `AI Training Model/Training Model.ipynb` and are wrote by Pytorch following the below steps:

### 1. Get and Preprocess Dataset
Load training data and test data from EMNIST dataset. Rotate the image by 90° then flip horizontally to make pictures' orientation consistent with how people normally view images. The tensor vectors are also normalized to the range of [-1, 1]. The batch size set 64.

### 2. Build the Model
The foward of convolutional neural network (CNN) is following: <br>
```
input
→ convolution layer 1 → batchnorm → ReLU
→ convolution layer 2 → batchnorm → ReLU
→ max-pooling
→ dropout
→ flatten
→ linear layer1 → ReLU
→ dropout
→ linear layer 2
→ output 
```

The kernel is 3×3 and the stride is 1.
`batchnorm` is batch normalization that stabilizes and accelerates training by normalizing layer activations within a mini-batch. <br>
Dropout prevents overfitting by randomly deactivating neurons during training. The first dropout in convolution layer is the probability of 25%, and the second one in linear layer is the probability of 50%.

### 3. Optimization
Each paramter is respectively learning rate = 1e-3 and weight decay(L2 regularization) = 1e-5. There is also a scheduler that to monitor the loss and reduce the learning rate if the loss does not decrease in 2 epochs.

### 4. Learning Epochs
The model runs 20 epochs and records each training loss, traning accuracy, validation (test) loss, and validation accuracy. In each training iteration, the model is updated through a backpropagation function.

### 5. Saving the Model
`AI Training Model/emnist_cnn.pth` stores the model parameters and the total number of classes.

# Website
The web contains an canvas that allows the user to draw a character on it. After that, users can press `predict` button to get the predicted result of the CNN model. If there is nothing on canvas, the error will pop up.<br>
write2predict-demo.streamlit.app/
