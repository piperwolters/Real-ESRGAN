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
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input tile')
    parser.add_argument('-model_path', type=str, help='Path to model weights', default='experiments/2S2_urban/models/net_g_225000.pth')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    # model 
    model = RRDBNet(num_in_ch=6, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
        print("loading....", model_path)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['params'])

    model.eval()

    input_path = '/data/piperw/inference_data/' + args.input + '/s2_condensed' 
    paths = sorted(glob.glob(os.path.join(input_path, '*/*.png')))

    paths = paths * 100

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
            batches.append(torch.stack(batch))
            batches_names.append(batch_names)
            batch, batch_names = [], []

    for i,batch in enumerate(batches):
        try:
            print("input:", batch.shape, " time:", time.perf_counter())
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(batch)
                    print("output:", output.shape, " time:", time.perf_counter())
                    output = torch.permute(output, (0, 2, 3, 1)).squeeze()
        except RuntimeError as error:
            print('Error', error)

        else:
            extension = 'png'
           
            print("saving...", time.perf_counter())
            for j,b in enumerate(output):
                basename, imgname = batches_names[i][j]

                os.makedirs(basename, exist_ok=True)
                save_path = os.path.join(basename, f'{imgname}.{extension}')

                #skimage.io.imsave(save_path, b.detach().numpy())
            print("done saving batch...", time.perf_counter())


if __name__ == '__main__':
    main()
