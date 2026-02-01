def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.float().to(device)
        loss = criterion(model(xb).squeeze(), yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)
