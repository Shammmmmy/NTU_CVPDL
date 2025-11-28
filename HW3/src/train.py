import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from diffusion_model import DiffusionModel, UNet
from torch.utils.tensorboard import SummaryWriter

class CusDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = [f for f in os.listdir(root) if f.endswith('.png')]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

def train(lr=2e-4, batch_size=64, epochs=50, n_step=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = './weights'
    os.makedirs(save_dir, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_path = os.path.join(script_dir, 'datasets')
    dataset = CusDataset(root=dataset_path, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = UNet().to(device)
    diffusion = DiffusionModel(n_step=n_step, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    best_loss = float('inf')
    writer = SummaryWriter('runs/ddpm_experiment')
    print("Start Training")

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar= tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x in pbar:
            x = x.to(device)

            t = torch.randint(0, n_step, (x.size(0),), device=device).long()
            
            loss = diffusion.p_loss(model, x, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Training Loss', avg_loss, epoch + 1)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_save_path = os.path.join(save_dir, "ddpm_best.pth")
            torch.save(model.state_dict(), best_save_path)
            print(f"Best model saved with loss: {best_loss:.6f}")

        save_path = os.path.join(save_dir, f"ddpm_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

    writer.close()
    print("Training Finished")


if __name__ == "__main__":
    lr = 2e-4
    batch_size = 64
    epochs = 50
    n_step = 1000

    train(lr=lr, batch_size=batch_size, epochs=epochs, n_step=n_step)    