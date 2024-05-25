
def showing_data(data_loader):
    for batch in data_loader:
        images, labels  = batch

        print(f"Shape of the first image in the batch: {images[0].shape}")

        first_image_channel = images[0][0].detach().cpu().numpy()
        print(f"First image channel: {first_image_channel}")

        break