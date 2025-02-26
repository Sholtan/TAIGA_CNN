
import numpy as np
import corsikadata
import simple_conv_net

from tqdm import tqdm_notebook

gamma_dataset = corsikadata.CorsikaData()
gamma_dataset.load("taiga607_st2b_0")


proton_dataset = corsikadata.CorsikaData()
proton_dataset.load("taiga623_st2b_0")



print(gamma_dataset.data.shape)
print(gamma_dataset.labels.shape)
print(proton_dataset.data.shape)
print(proton_dataset.labels.shape)

input_data = np.concatenate((gamma_dataset.data, proton_dataset.data), axis=0)
input_labels = np.concatenate((gamma_dataset.labels, proton_dataset.labels), axis=0)

print(input_data.shape)
print(input_data.dtype)
print(input_labels.shape)
print(input_labels.dtype)

input_data = np.expand_dims(input_data, axis = 1)
print(input_data.shape)

rng = np.random.default_rng(42)
permut = rng.permutation(len(input_data))
input_data = input_data[permut]
input_labels = input_labels[permut]


train_portion = int(0.8 * len(input_data))

train_data = input_data[:train_portion]
train_labels = input_labels[:train_portion]
test_data = input_data[train_portion:-1]
test_labels = input_labels[train_portion:-1]

print("train_data, train_labels, test_data, test_labels shapes")
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
print()

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=False, num_workers=2)
train_labels_loader = torch.utils.data.DataLoader(train_labels, batch_size=100, shuffle=False, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)
test_labels_loader = torch.utils.data.DataLoader(test_labels, batch_size=100, shuffle=False, num_workers=2)


net = simple_conv_net.SimpleConvNet()


def accuracy_on_test(model, dataloader, labelsloader):
    correct = 0
    for data, labels in zip(dataloader, labelsloader):
        y_pred = net(data.to(device))
        y_pred = y_pred.reshape(len(labels))
        y_pred = y_pred>0.5
        c = y_pred.cpu()==labels
        correct += c.sum()
    return correct/test_labels.shape[0]

losses = []
accuracies = []

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1)

# итерируемся
for epoch in tqdm_notebook(range(10)):
    running_loss = 0.0
    i = 0
    for data, labels in zip(train_loader, train_labels_loader):
        net.my_optimizer.zero_grad()
        
        y_pred = net(data.to(net.device))
        y_pred = y_pred.reshape(len(labels))
        
        labels = labels.to(torch.float32)   
        y_pred = y_pred.to(torch.float32)   
        
        loss = loss_fn(y_pred, labels.to(net.device))
        loss.backward()
        net.my_optimizer.step()
        
        running_loss += loss.item()
        if i % 300 == 299:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            losses.append(running_loss)
            accuracies.append(accuracy_on_test(net, test_loader, test_labels_loader))
            running_loss = 0.0
        i+=1
    ax.clear()
    ax.plot(np.arange(len(losses)), losses)
    plt.show()
print('Обучение закончено')