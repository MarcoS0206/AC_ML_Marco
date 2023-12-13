import torch
import wandb

from dataset import test_loader, train_loader
from resnet import model18, criterion, optimizer, device

epoch_Loss = 0

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:  # Prints every 50 mini-batches
            print(f'Batch {i + 1}, Loss: {running_loss / 50:.3f}')
            running_loss = 0.0

    epoch_Loss = running_loss / 256
    print(f'TOTAL EPOCH LOSS: {epoch_Loss}')
    return epoch_Loss

test_accuracy = 0

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    test_accuracy = accuracy
    print(f'Accuracy on the test set: {100 * accuracy:.2f}%')
    return test_accuracy






if __name__ == '__main__':
    
    wandb.login()

    wandb.init(
    # set the wandb project where this run will be logged
    project="ResNet19-CIFAR10-test",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "RESNET18 +1 layer",
    "dataset": "CIFAR-10",
    "epochs": 30,
    }
)
    num_epochs = 30

    current_epoch = 0

    print("USING DEVICE: ",device)

    # initialize Step LR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training the model
    for epoch in range(num_epochs):
        current_epoch += 1
        print("")
        print("Epoch: ",current_epoch)
        #train_loop(tr_loader, model18, criterion, optimizer)
        
    
        Loss = train(model18, train_loader, criterion, optimizer, device)
        Acc = test(model18, test_loader, device)
        scheduler.step()
        LearnRate = scheduler.get_last_lr()[0]
        print("LR: ",LearnRate)
        wandb.log({"Accuracy":Acc, "Loss": Loss, "LearningRate":LearnRate})
        

    print('Finished Training')
    wandb.finish()
    
    # Testing the model
    test(model18, test_loader, device)