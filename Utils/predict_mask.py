class ImageDataset(Dataset):
    def __init__(self, folder_path, image_height, image_width):
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[1.0])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")
        return self.transform(image), image_path


def load_checkpoint(filepath, model, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

def preprocess_image(image_path, height, width):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict_mask(model, input_image, device):
    input_image = input_image.to(device)
    with torch.no_grad():
        prediction = model(input_image)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    return prediction.cpu().numpy()

def apply_mask_to_image(image, mask):
    mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    image = image.convert("RGBA")
    mask_rgba = mask.convert("RGBA")
    masked_image = Image.blend(image, mask_rgba, alpha=0.5)
    return masked_image.convert("RGB")

def save_image(image, output_path):
    image.save(output_path)

def process_images(model, device, data_loader, image_height, image_width, output_dir):
    for input_images, image_paths in data_loader:
        input_images = input_images.to(device)
        predicted_masks = predict_mask(model, input_images, device)

        for i in range(predicted_masks.shape[0]):
            predicted_mask = predicted_masks[i, 0, :, :]
            original_image = Image.open(image_paths[i]).convert("L").resize((image_width, image_height))
            masked_image = apply_mask_to_image(original_image, predicted_mask)
            save_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_paths[i]))[0] + ".jpg")
            save_image(masked_image, save_path)

def main(folder_path, model_path, image_height, image_width, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=1, out_channels=1).to(device)
    load_checkpoint(model_path, model, device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = ImageDataset(folder_path, image_height, image_width)
    data_loader = DataLoader(dataset, batch_size=20, num_workers=mp.cpu_count())

    process_images(model, device, data_loader, image_height, image_width, output_dir)

if __name__ == "__main__":
    folder_path = "new_dataset/train/images"
    model_path = "best_model.pth.tar"
    IMAGE_HEIGHT = 380
    IMAGE_WIDTH = 540
    output_dir = "applied_masks"

    main(folder_path, model_path, IMAGE_HEIGHT, IMAGE_WIDTH, output_dir)
