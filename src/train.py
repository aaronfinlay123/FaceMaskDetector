import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import NeuralNet

#download CIFAR 10 data
trainset = torchvision.datasets.CIFAR10(
    root="./data",
    Train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
trainloader = torch.utils.Data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

if __name__ == '__main__':
    #define convolutional neural network
    net = NeuralNet()

    #set up pytorch loss function / optimization
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #train the network
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #unpack the data
            inputs, labels = data

            #zero the parameter gradients
            optimizer.zero_grad()

            #forward propagation, backward propagation, optimization
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #print stats
            running_loss += loss.item()
            if i % 2000 == 1999:
                loss = running_loss / 2000
                print(f"epoch={epoch+1}, batch={i + 1:5}: loss {loss:.2f}")
                running_loss = 0.0
    print("Finished Training Data")