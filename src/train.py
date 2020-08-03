import argparse
import math
import random
from os import path, makedirs
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import skimage.io
from skimage.transform import rescale, resize
from tqdm import tqdm
from model import singan

parser = argparse.ArgumentParser(description='Train SinGAN.')
parser.add_argument('target', help='filename of the target image (relative path of project root).')

ROOT_PATH = str(Path(path.abspath(path.dirname(__file__))).parent)

def train(input_image_path: str, trained_folder: str):
    seed = random.randint(1, 10000)
    print(f'Seed: {seed}')
    torch.manual_seed(seed)

    mseloss = nn.MSELoss()

    normalize = lambda x: (x - 0.5) * 2
    denormalize = lambda x: (x + 1) / 2

    source_img = skimage.io.imread(input_image_path)
    init_scale = min(250 / max(source_img.shape[0], source_img.shape[1]), 1)
    num_scales = math.ceil((math.log(math.pow(25 / (min(source_img.shape[0], source_img.shape[1])), 1), 0.75))) + 1
    fin_scale = num_scales - math.ceil(math.log(min(250, max(source_img.shape[0], source_img.shape[1])) / max(source_img.shape[0], source_img.shape[1]), 0.75))

    source_img_init = rescale(source_img, init_scale, anti_aliasing=True, preserve_range=True, multichannel=True)
    scale_factor = math.pow(25 / min(source_img_init.shape[0], source_img_init.shape[1]), 1 / fin_scale)
    fin_scale = num_scales - math.ceil(math.log(min(250, max(source_img_init.shape[0], source_img_init.shape[1])) / max(source_img_init.shape[0], source_img_init.shape[1]), 0.75))

    print(f'Scale (goal): {fin_scale}')

    source_img_pyramids = []

    for i in range(fin_scale + 1):
        curr_scale = math.pow(scale_factor, fin_scale - i)
        curr_img = rescale(source_img, curr_scale, anti_aliasing=True, preserve_range=True, multichannel=True).transpose(2, 0, 1) / 255
        curr_img = np.clip(normalize(curr_img), -1, 1)[0:3,:,:]
        source_img_pyramids.append(torch.from_numpy(curr_img).float().cuda())

    generators = []
    fixed_noises = []
    noise_amplifiers = []

    out_chan_prev = None

    for i in range(fin_scale + 1):
        torch.cuda.empty_cache()

        print(f'Scale: {i}')
        out_chan_init = min(32 * pow(2, i // 4), 128)
        out_chan_min = out_chan_init

        curr_G = singan.Generator(out_chan_init, out_chan_min)
        curr_G.apply(singan.init_weights)
        curr_G.cuda()

        curr_D = singan.Discriminator(out_chan_init, out_chan_min)
        curr_D.apply(singan.init_weights)
        curr_D.cuda()

        if out_chan_init == out_chan_prev:
            prev_i = i - 1
            curr_G.load_state_dict(torch.load(path.join(trained_folder, f'{prev_i}_generator.bin')))
            curr_D.load_state_dict(torch.load(path.join(trained_folder, f'{prev_i}_discriminator.bin')))

        out_chan_prev = out_chan_init

        target_image = source_img_pyramids[i]
        image_height, image_width = target_image.shape[1], target_image.shape[2]
        padding = nn.ZeroPad2d(((3 - 1) * 5) // 2)

        if i == 0:
            fixed_noise = torch.randn(1, 1, image_height, image_width, device='cuda')
            fixed_noise = padding(fixed_noise.expand(1, 3, image_height, image_width))
            prev_rec_fake = padding(torch.full([1, 3, image_height, image_width], 0, dtype=torch.float32, device='cuda'))
            noise_amp = 1
        else:
            fixed_noise = padding(torch.full([1, 3, image_height, image_width], 0, dtype=torch.float32, device='cuda'))
            prev_rec_fake = apply_pyramid(generators, fixed_noises, source_img_pyramids, padding, noise_amplifiers, True)
            noise_amp = 0.1 * torch.sqrt(mseloss(torch.unsqueeze(target_image, 0), prev_rec_fake))
            prev_rec_fake = padding(prev_rec_fake)

        optimizer_dis = optim.Adam(curr_D.parameters(), lr=.0005, betas=(0.5, 0.999))
        optimizer_gen = optim.Adam(curr_G.parameters(), lr=.0005, betas=(0.5, 0.999))
        scheduler_dis = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_dis, milestones=[1600], gamma=0.1)
        scheduler_gen = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_gen, milestones=[1600], gamma=0.1)

        for _ in tqdm(range(2000)):
            if i == 0:
                random_noise = padding(torch.randn(1, 1, image_height, image_width, device='cuda').expand(1, 3, image_height, image_width))
            else:
                random_noise = padding(torch.randn(1, 3, image_height, image_width, device='cuda'))

            for _ in range(3):
                curr_D.zero_grad()

                real_out = curr_D(torch.unsqueeze(target_image, 0)).cuda()
                real_loss = real_out.mean() * -1
                real_loss.backward(retain_graph=True)

                fake_image = padding(apply_pyramid(generators, fixed_noises, source_img_pyramids, padding, noise_amplifiers))
                fake_w_noise_applied = noise_amp * random_noise + fake_image

                generated_fake = curr_G(fake_w_noise_applied.detach(), fake_image)
                fake_out = curr_D(generated_fake.detach())
                fake_loss = fake_out.mean()
                fake_loss.backward(retain_graph=True)

                gp = gradient_penalty(curr_D, torch.unsqueeze(target_image, 0), generated_fake)
                gp.backward()

                optimizer_dis.step()

            for _ in range(3):
                curr_G.zero_grad()

                disc_out = curr_D(generated_fake)
                loss = disc_out.mean() * -1
                loss.backward(retain_graph=True)
                noise = noise_amp * fixed_noise if i == 0 else prev_rec_fake
                rec_loss = 10 * mseloss(curr_G(noise.detach(), prev_rec_fake), torch.unsqueeze(target_image, 0))
                rec_loss.backward(retain_graph=True)

                optimizer_gen.step()

            scheduler_dis.step()
            scheduler_gen.step()

        image_to_save = (denormalize(generated_fake).squeeze().clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        skimage.io.imsave(path.join(trained_folder, f'{i}_fake_sample.png'), image_to_save)

        torch.save(curr_G.state_dict(), path.join(trained_folder, f'{i}_generator.bin'))
        torch.save(curr_D.state_dict(), path.join(trained_folder, f'{i}_discriminator.bin'))
        if i == 0:
            torch.save(fixed_noise, path.join(trained_folder, f'{i}_fixed_noise.bin'))

        for param in curr_G.parameters():
            param.requires_grad_(False)
        curr_G.eval()

        generators.append(curr_G)
        fixed_noises.append(fixed_noise)
        noise_amplifiers.append(noise_amp)

    torch.save(generators, path.join(trained_folder, 'generators.bin'))
    torch.save(source_img_pyramids, path.join(trained_folder, 'source_images.bin'))
    torch.save(noise_amplifiers, path.join(trained_folder, 'noise_amplifiers.bin'))

def apply_pyramid(generators, fixed_noises, source_images, padding, noise_amplifiers, rec_image=False):
    result_image = torch.full([1, 3, source_images[0].shape[1], source_images[0].shape[2]], 0, dtype=torch.float32, device='cuda')
    if len(generators) == 0:
        return result_image
    normalize = lambda x: (x - 0.5) * 2
    denormalize = lambda x: (x + 1) / 2
    for scale in range(len(generators)):
        if rec_image:
            generated_noise = fixed_noises[scale]
        elif scale == 0:
            generated_noise = torch.randn(1, 1, source_images[scale].shape[1], source_images[scale].shape[2], device='cuda')
            generated_noise = padding(generated_noise.expand(1, 3, generated_noise.shape[2], generated_noise.shape[3]))
        else:
            generated_noise = padding(torch.randn(1, 3, source_images[scale].shape[1], source_images[scale].shape[2], device='cuda'))
        result_image = padding(result_image[:,:,0:source_images[scale].shape[1],0:source_images[scale].shape[2]])
        image_w_noise = noise_amplifiers[scale] * generated_noise + result_image
        result_image = generators[scale](image_w_noise.detach(), result_image)
        result_image = (denormalize(result_image).squeeze().clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        result_image = torch.from_numpy(resize(result_image, (source_images[scale+1].shape[1], source_images[scale+1].shape[2]), anti_aliasing=True, preserve_range=True)).float().cuda()
        result_image = normalize(result_image.permute(2, 0, 1) / 255).clamp(-1, 1)[0:3,:,:]
        result_image = torch.unsqueeze(result_image, 0)
    return result_image

def gradient_penalty(curr_d, real_out, fake_out):
    alpha = torch.rand(1, 1).expand(real_out.size()).cuda()
    ip = (alpha * real_out + ((1 - alpha) * fake_out)).cuda()
    ip = torch.autograd.Variable(ip, requires_grad=True)
    dis_ip = curr_d(ip)
    grad = torch.autograd.grad(outputs=dis_ip, inputs=ip, grad_outputs=torch.ones(dis_ip.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return 0.1 * ((grad.norm(2, dim=1) - 1) ** 2).mean()

if __name__ == "__main__":
    args = parser.parse_args()
    TARGET_IMAGE_PATH = path.join(ROOT_PATH, args.target)
    TARGET_IMAGE_NAME = path.splitext(path.basename(TARGET_IMAGE_PATH))[0]
    TRAINED_FOLDER = path.join(ROOT_PATH, 'trained', TARGET_IMAGE_NAME)

    if not path.isdir(TRAINED_FOLDER):
        makedirs(TRAINED_FOLDER)
    if not path.isfile(TARGET_IMAGE_PATH):
        raise RuntimeError('Target image was not found.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device was not found.')
    train(TARGET_IMAGE_PATH, TRAINED_FOLDER)
