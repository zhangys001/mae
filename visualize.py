import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models_mae

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    img = image * std + mean
    img = torch.clip(img * 255, 0, 255).int()
    plt.imshow(img)
    plt.title(title, fontsize=14)
    plt.axis('off')

def visualize(checkpoint_path, model_name='mae_vit_base_patch4', mask_ratio=0.75, n_imgs=4):
    dataset = datasets.STL10(root='./data', split='test', download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(96),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
                              ]))

    model = models_mae.__dict__[model_name](norm_pix_loss=False)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    torch.manual_seed(42)
    fig, axes = plt.subplots(n_imgs, 4, figsize=(12, 3 * n_imgs))

    for i in range(n_imgs):
        idx = torch.randint(0, len(dataset), (1,)).item()
        img, label = dataset[idx]
        x = img.unsqueeze(0)

        with torch.no_grad():
            loss, y, mask = model(x, mask_ratio=mask_ratio)
            y = model.unpatchify(y)

        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
        mask = model.unpatchify(mask)

        im_masked = x * (1 - mask)
        im_paste = x * (1 - mask) + y * mask

        for j, (tensor, title) in enumerate([
            (x[0], f'Original (label={label})'),
            (im_masked[0], 'Masked'),
            (y[0], 'Reconstructed'),
            (im_paste[0], 'Recon + visible')
        ]):
            ax = axes[i, j]
            img_np = tensor.permute(1, 2, 0).numpy()
            img_np = img_np * std + mean
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            ax.imshow(img_np)
            ax.set_title(title, fontsize=11)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved to reconstruction.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./output_dir/checkpoint-199.pth', help='path to checkpoint')
    parser.add_argument('--model', default='mae_vit_base_patch4')
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--n_imgs', type=int, default=4)
    args = parser.parse_args()
    visualize(args.checkpoint, args.model, args.mask_ratio, args.n_imgs)
