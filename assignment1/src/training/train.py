def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for image, labels in dataloader:
        images = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits =model(images)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        prediction = logits.argmax(dim=1)
        total_correct += (prediction == labels).sum().item()

    average_loss = total_loss / total_samples
    accuracy = total_correct /total_samples

    return {"loss": average_loss, "accuracy": accuracy}