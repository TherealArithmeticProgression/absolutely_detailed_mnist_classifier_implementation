"""This simple code block trains a neural network and tests it's accuracy on the torchvision
    MNIST dataset.
"""

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun
from torch.utils.data import DataLoader

# torch-vision imports
import torchvision.datasets as datasets
import torchvision.transforms as transforms

''' What do fc1/fc2/fc3 indicate?
    
-   Fully connected layers. Used in almost all ML tasks. 
    Useful for coming up with features (ie, neurons dedicated to a specific subset of features).
    A deep model with narrow layers (e.g., 10 layers of 10 neurons each) has far fewer parameters and a lower risk of overfitting than a single massive fully connected layer (e.g., 784 to 10,000 neurons).
    
    what does def forward define?
-   Here, what it does is: output_of_fc(1/2)=weights_of_fc_(1/2)*(input_of_fc_(1/2))+bias_vector_of_fc_(1/2).
    Then, any negative number in the matrix is replaced with 0. 
    
    why is relu (rectified linear unit) used here?
-   ReLU is an activation function (like sigmoid/tanh/elu etc)
    If a neuron outputs -50.0 (meaning "I am absolutely, definitely sure there are no eyes here"), 
    ReLU steps in and says: "I don't care about your negative certainty. 
    Zero evidence is zero evidence. Output a 0."
    Although, blocking all negative values blindly creates the "Dying ReLU" problem. 
    If a neuron outputs negative values across your entire dataset, its gradient becomes exactly zero. 
    It dies, stops learning completely, and the optimizer can never wake it up.
    
    why did we have the `self.fc(x)` statements at all?
    output_of_fc(1/2/3)=weights_of_fc_(1/2/3)*(input_of_fc_(1/2/3))+bias_vector_of_fc_(1/2/3).
    
'''
#make a nn class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28*28, 14*14)
        self.fc2 = nn.Linear(14*14, 7*7)
        self.fc3 = nn.Linear(7*7, 10)

    def forward(self, x):
        x=fun.relu(self.fc1(x))
        x=fun.relu(self.fc2(x))
        x=self.fc3(x)
        return x

#     set hyperparameters:

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 28*28
output_size = 10
batch_size = 150
num_epochs= 15
num_classes = 10
learning_rate = 0.001

'''
    What does the dataloader refer to?
    A dataset is made for people. It is arranged in a human-friendly format (like a table, or images arranged in a serial order etc).
    A model isn't a human. It expects the data in a particular manner. So the dataloader 
    takes the dataset and modifies it for the model. 
    batching: a model can't wait to be fed one image at a time. It can glimpse and learn pretty fast.
    so the images are bundled in a batch.  This as well done by the dataloader.
    backstage loading: can't wait for model to finish batch 1 then take images and bundle them into batch 2.
    So there are background workers batching the images together while model is looking at the first batch.
    shuffling: what if model also memorizes the order in which the images appear? (all images of raccoons are together, followed by salamander image etc
    the model might end up remembering the order, so the images are shuffled by the dataloader.
    
    Why do you need two distinct types of dataloaders?
-   They're both loading different datasets, and shuffle parameter changes. could keep one as well, this seems simpler/more "intuitive".
'''

#     create the test/train dataset and dataloaders

train_dataset=datasets.MNIST(root='./dataset',train=True, transform=transforms.ToTensor(), download=True)
test_dataset=datasets.MNIST(root='./dataset',train=False, transform=transforms.ToTensor(), download=True)
train_loader=DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
test_loader=DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=False)


'''
    What does cross entropy loss measure?
-   High Entropy: 
    If your model looks at a cat image and outputs [Cat: 33%, Dog: 33%, Bird: 34%], its answer is pure chaos and random guessing. 
    It has high entropy.
    Low Entropy: 
    If your model outputs [Cat: 98%, Dog: 1%, Bird: 1%], it is highly confident and orderly. 
    It has low entropy.
    The loss function compares (crosses) two different probability distributions:
    The Model's Guess: [Cat: 40%, Dog: 50%, Bird: 10%]
    The True Answer (Ground Truth): [Cat: 100%, Dog: 0%, Bird: 0%]
    It measures the divergence between two distributions. If your model is 100% confidently wrong (e.g., predicting 100% Dog when the true label is Cat), its internal entropy is 0 (perfectly orderly, zero chaos), but your Cross-Entropy loss will approach infinity. 
    It punishes incorrect confidence, not just "chaos."
    
    Why is adam optimizer used here at all?
    Best for the mid sized three layer network. It is (almost) an industry-wide practise.
    An optimizer is responsible for changing the weights according to evaluation it recieves (cross entropy loss, for eg).
    
    Why did we reshape the data? 
    A single MNIST image is a 2D grid: 28 pixels wide by 28 pixels high.
    Layer fc1 is a flat line of input slots. It has no concept of "up", "down", "left", or "right".
    Reshaping unrolls that 28x28 square grid into a single, long line of 784 pixels (1 * 28 * 28 = 784).
    (In frameworks like PyTorch or TensorFlow, -1 is a special keyword shortcut.)
    (The creators of PyTorch/TensorFlow explicitly hardcoded -1 as the designated flag for "automatic dimension inference.")    
'''
model=NeuralNetwork().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    print("Epoch: ", epoch+1)
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.to(device=device)
        target=target.to(device=device)
        data=data.reshape(data.shape[0],-1)
        score=model(data)
        loss=criterion(score,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''
    What is the function of 
    
    : model.eval()??
    Ensures that the model is in testing/validation mode. 
    Makes dropout(randomly switch off neurons to prevent overfitting on a couple of neurons) and batchnorm(batch normalization:  Softmax, used to calculate final probabilities, only cares about relative differences. But we use BatchNorm to keep numbers little because computers are fragile. Keeping magnitudes small prevents infinity errors and keeps the gradients flowing so Adam can actually do his job.)
    stop and freeze, respectively, for a while.
    In Training Mode: 
    BatchNorm has to look at your evaluation batch, calculate the exact average, calculate the variance, and then scale the numbers.
    That is a lot of heavy math overhead.
    In Evaluation Mode: 
    BatchNorm skips all those calculations entirely.
    It just grabs the historical average it memorized during training and instantly applies it. 
    It behaves like a simple, fast multiplier, which actually speeds up your response time
    
    : torch.no_grad()?
    Don't make gradient computations. 
    Gradient computations are only used to evaluate the model's performance and make corresponding changes in the weights.
    We aren't changing weights while checking accuracy, rather we're asking "do we really need to change the weights?"
    
    : model.train()?
    Restart all the stuff that stopped otherwise both the dropout and batchnorm will stay halted/frozen in their current state.
    Switch these on, now we're training again.
'''
def check_accuracy(loader, modl):
    total=0
    correct=0
    modl.eval()
    with torch.no_grad():
        for dta, trgt in loader:
            dta=dta.reshape(dta.shape[0],-1).to(device=device)
            trgt=trgt.to(device=device)
            score_acc=modl(dta)

            _,prediction=torch.max(score_acc.data,1)
            total+=trgt.size(0)
            correct+=(prediction==trgt).sum().item()
        modl.train()
    return f'The accuracy obtained is: {(correct*100/total):.2f}'


print("**Training**\n" + check_accuracy(train_loader, model))
print("**Testing**\n" + check_accuracy(test_loader, model))
