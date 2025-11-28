import os
import torch
from tqdm import tqdm
from diffusion_model import DiffusionModel, UNet
from torchvision.utils import save_image, make_grid

def generate_visualization(model, diffusion, img_size, channels, report_img):
    visual_size = 8

    with torch.no_grad():
        final_img, intermediate_imgs = diffusion.sample(model, visual_size, img_size, channels)
    step = intermediate_imgs
    all_steps = step
    stack_imgs = torch.stack(all_steps)
    stack_imgs = stack_imgs.view(-1, channels, img_size, img_size)
    grid = make_grid(stack_imgs, nrow=visual_size, padding=2, pad_value=0)
    save_image(grid, report_img)
    print(f"Saved visualization image: {report_img} (8x8 Grid)")

def generate_images(model, diffusion, total_images, img_size, channels):
    print(f"Generating {total_images} images...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    count = 0
    pbar = tqdm(total=total_images)

    while count < total_images:
        batch_size = min(100, total_images - count)
        with torch.no_grad():
            sampled_imgs, _ = diffusion.sample(model, batch_size, img_size, channels)

        for i in range(batch_size):
            count += 1
            save_image(sampled_imgs[i], os.path.join(output_dir, f"{count:05d}.png"))
            
        pbar.update(batch_size)
    pbar.close()
    print(f"Generated {total_images} images in directory: {output_dir}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights_path = './weights/ddpm_best.pth'
    output_dir = 'img_M11315025'
    report_img = 'diffusion_process_report_M11315025.png'
    n_step = 1000
    total_images = 10000
    img_size = 28
    channels = 3

    model = UNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    diffusion = DiffusionModel(n_step=n_step, device=device)

    generate_visualization(model, diffusion, img_size, channels, report_img)
    generate_images(model, diffusion, total_images, img_size, channels)