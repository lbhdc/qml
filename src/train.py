import torch.optim as optim
import torch.nn as nn


def train(model, dataset, *, epochs=20, learning_rate=0.0001, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.NLLLoss()
    loss_list = []

    model.train()
    for epoch in range(epochs):

        total_loss = []
        for (data, target) in dataset:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        loss_list.append(sum(total_loss) / len(total_loss))

        if verbose:
            progress = int(100. * (epoch + 1) / epochs)
            tloss = round(loss_list[-1], 5)
            print(f"Training: Progress [{progress}%] Loss [{tloss}]")

    return model
