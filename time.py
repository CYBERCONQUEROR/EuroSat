import time

start_time = time.time()

# Training loop
for epoch in range(20):
    model.train()

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] completed")

end_time = time.time()

total_time = end_time - start_time

print(f"\nTotal Training Time: {total_time:.2f} seconds")
print(f"Total Training Time: {total_time/60:.2f} minutes")