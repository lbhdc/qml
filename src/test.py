import torch


def test(model, dataset, *, verbose=True):
    loss_func = torch.nn.NLLLoss()
    model.eval()
    total_loss = []

    with torch.no_grad():
        correct = 0
        for (data, target) in dataset:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = loss_func(output, target)
            total_loss.append(loss)

        if verbose:
            tloss = round(float(sum(total_loss) / len(total_loss)), 5)
            tacc = round(float((correct / len(dataset) * 100)), 5)
            print(f"Result: Loss [{tloss}] Accuracy [{tacc}]")

    return model
