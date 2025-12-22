import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module): 
    def __init__(self, num_classes=47, dropout_conv=0.25, dropout_fc=0.5): # 47 classes for balance split
        super(CNN,self).__init__()
        # convolution 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32) # Add BatchNorm after conv1

        # convolution 2
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64) # Add BatchNorm after conv2

        self.pool = nn.MaxPool2d(2, 2)
        # dropout layers: spatial dropout for conv outputs, regular dropout for FC
        self.dropout_conv = nn.Dropout2d(dropout_conv)
        self.dropout = nn.Dropout(dropout_fc)

        # determine flatten dimention
        self.flatten_dim = self._get_flatten_dim()
        # get full connect layer latter
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    # calculate flatten features
    def _get_flatten_dim(self):
        with torch.no_grad():
            # dummy image
            x = torch.zeros(1, 1, 28, 28)
            # conv1 -> relu -> conv2 -> relu -> pool -> dropout
            # Apply BatchNorm before ReLU in the calculation of flatten_dim
            x= self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))

            x = self.dropout_conv(x)
            # get num of features
            return x.numel()
    
    def forward(self, x):
        # conv layers with pooling and conv-dropout
        # Apply BatchNorm after conv and before ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.dropout_conv(x)
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        # apply dropout between FC layers (only active during training)
        x = self.dropout(x)

        return self.fc2(x)

def predict(input):
    """
    predict the image by the trained model
    """
    # load hyperparameters and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(num_classes = 47).to(device)

    ckpt = torch.load('./AI Training Model/emnist_cnn.pth', map_location=device)
    model.load_state_dict(ckpt["model_state"])
    classes = ckpt['classes']

    # set model to eval mode
    model.eval()

    # get the index with the highest probability 
    with torch.no_grad():
        output = model(input.to(device))
        pred = output.argmax(dim = 1).item()

    return classes[pred]
