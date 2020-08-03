import argparse
import math
from os import path, makedirs
from pathlib import Path
import torch
from torch import nn
import skimage.io
from skimage.transform import resize, rescale
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate images from trained model.')
parser.add_argument('target', help='filename of the target image (relative path of project root).')
parser.add_argument('--num', type=int, default=10, help='number of images to generate (default 10).')

ROOT_PATH = str(Path(path.abspath(path.dirname(__file__))).parent)

def generate_fake_images(input_image_path: str, trained_folder: str, generated_folder: str, image_num: int):
    source_img = skimage.io.imread(input_image_path)
    init_scale = min(250 / max(source_img.shape[0], source_img.shape[1]), 1)
    num_scales = math.ceil((math.log(math.pow(25 / (min(source_img.shape[0], source_img.shape[1])), 1), 0.75))) + 1

    source_img_init = rescale(source_img, init_scale, anti_aliasing=True, preserve_range=True, multichannel=True)
    fin_scale = num_scales - math.ceil(math.log(min(250, max(source_img_init.shape[0], source_img_init.shape[1])) / max(source_img_init.shape[0], source_img_init.shape[1]), 0.75))

    try:
        generators = torch.load(path.join(trained_folder, 'generators.bin'))
        source_img_pyramids = torch.load(path.join(trained_folder, 'source_images.bin'))
        noise_amplifiers = torch.load(path.join(trained_folder, 'noise_amplifiers.bin'))
    except:
        print('Trained model was not found.')
        exit(1)

    normalize = lambda x: (x - 0.5) * 2
    denormalize = lambda x: (x + 1) / 2

    for num in tqdm(range(image_num)):
        prev_target_image = None
        for scale in range(len(generators)):
            pad = nn.ZeroPad2d(((3 - 1) * 5) // 2)
            height, width = source_img_pyramids[scale].shape[1], source_img_pyramids[scale].shape[2]

            if scale == 0:
                target_image = pad(torch.full(source_img_pyramids[0].shape, 0, device='cuda'))
                noise_image = torch.randn(1, 1, height, width, device='cuda')
                noise_image = pad(noise_image.expand(1, 3, height, width))
            else:
                target_image = prev_target_image
                target_image = (denormalize(target_image).squeeze().clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                target_image = torch.from_numpy(resize(target_image, (source_img_pyramids[scale].shape[1], source_img_pyramids[scale].shape[2]), anti_aliasing=True, preserve_range=True)).float().cuda()
                target_image = pad(normalize(target_image.permute(2, 0, 1) / 255).clamp(-1, 1)[0:3,:,:])
                noise_image = pad(torch.randn(1, 3, height, width, device='cuda'))

            target_image = torch.unsqueeze(target_image, 0)
            noise_current = noise_amplifiers[scale] * noise_image + target_image
            generated_image = generators[scale](noise_current.detach(), target_image)
            prev_target_image = generated_image

            if scale == fin_scale:
                image_to_save = (denormalize(generated_image).squeeze().clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                skimage.io.imsave(path.join(generated_folder, f'generated_{num}.png'), image_to_save)

if __name__ == "__main__":
    args = parser.parse_args()
    TARGET_IMAGE_PATH = path.join(ROOT_PATH, args.target)
    TARGET_IMAGE_NAME = path.splitext(path.basename(TARGET_IMAGE_PATH))[0]
    TRAINED_FOLDER = path.join(ROOT_PATH, 'trained', TARGET_IMAGE_NAME)
    GENERATED_FOLDER = path.join(ROOT_PATH, 'generated', TARGET_IMAGE_NAME)

    if not path.isdir(TRAINED_FOLDER):
        raise RuntimeError('Trained model was not found.')
    if not path.isdir(GENERATED_FOLDER):
        makedirs(GENERATED_FOLDER)
    if not path.isfile(TARGET_IMAGE_PATH):
        raise RuntimeError('Target image was not found.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device was not found.')
    generate_fake_images(TARGET_IMAGE_PATH, TRAINED_FOLDER, GENERATED_FOLDER, args.num)
