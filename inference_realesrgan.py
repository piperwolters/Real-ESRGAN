import argparse
import time
import cv2
import torch
import numpy as np
import skimage.io
import glob
import os
import torchvision
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer

totensor = torchvision.transforms.ToTensor()

def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input path to directory containing tiles.')
    parser.add_argument('-model_path', type=str, help='Path to model weights', default='experiments/2S2_urban/models/net_g_225000.pth')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    device = torch.device("cuda")

    # model 
    model = RRDBNet(num_in_ch=6, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model = model.to(device)
    netscale = 4

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
        print("loading....", model_path)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['params_ema'])

    model.eval()

    input_path = args.input
    paths = []
    for tile in os.listdir(input_path):
        paths.extend(sorted(glob.glob(os.path.join(input_path, tile, 's2_condensed', '*/*.png'))))
    print("Number of images...", len(paths))

    batches_names, batch_names = [], []  # list of list of [basename, imgname]
    batches = []
    batch = []
    for idx, path in enumerate(paths):

        basename = path.split('/')
        basename = '/'.join(basename[:-1])
        basename = basename.replace('s2_condensed', 'inference')

        imgname, extension = os.path.splitext(os.path.basename(path))

        img = skimage.io.imread(path)
        img = np.reshape(img, (-1, 32, 32, 3))
        s2_chunks = [totensor(im) for im in img]
        img = torch.cat(s2_chunks, 0)

        batch.append(img)
        batch_names.append([basename, imgname])
        if len(batch) == 64:
            batches.append(torch.stack(batch).to(device))
            batches_names.append(batch_names)
            batch, batch_names = [], []

    extension = 'png'
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):

            for i,btch in enumerate(batches):
                try:
                    output = model(btch)
                    output = torch.permute(output, (0, 2, 3, 1))
                except RuntimeError as error:
                    print('Error', error)

                else:
                    for j,b in enumerate(output):
                        basename, imgname = batches_names[i][j]

                        os.makedirs(basename, exist_ok=True)
                        save_path = os.path.join(basename, f'{imgname}.{extension}')

                        skimage.io.imsave(save_path, b.detach().cpu().numpy())


if __name__ == '__main__':
    main()
