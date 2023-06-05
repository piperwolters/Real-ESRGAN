import os
import glob
import torch
import random
import torchvision
import skimage.io
import numpy as np
from PIL import Image
from torchvision.transforms import functional as trans_fn

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import realesrgan.data.util as Util


totensor = torchvision.transforms.ToTensor()

@DATASET_REGISTRY.register()
class RealESRGANPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RealESRGANPairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None


        # Adding code here to deal with specific satellite imagery input. 
        # Uncomment below code if you want to run original code / dataset structure.

        datatype = 's2'
        self.datatype = datatype
        self.n_s2_images = 18 #n_s2_images
        self.output_size = 128 #output_size
        self.max_tiles = -1 #max_tiles
        specify_val = True

        self.split = opt['phase']
        print("self.split ==", self.split)
        if self.split == 'train':
            dataroot = '/data/piperw/urban_set/'
        elif self.split == 'val':
            dataroot = '/data/piperw/held_out_set/'

        # Paths to the imagery.
        self.s2_path = os.path.join(dataroot, 's2_condensed')
        if self.output_size == 512:
            self.naip_path = os.path.join(dataroot, 'naip')
        elif self.output_size == 128:
            self.naip_path = os.path.join(dataroot, 'naip_128')
        elif self.output_size == 64:
            self.naip_path = os.path.join(dataroot, 'naip_64')
        elif self.output_size == 32:
            self.naip_path = os.path.join(dataroot, 'naip')
        else:
            print("WARNING: output size not supported yet.")

        # Load in the list of naip images that we want to use for val.
        self.val_fps = []
        if specify_val:
            val_fps_f = open('held_out.txt')
            val_fps = val_fps_f.readlines()
            for fp in val_fps:
                fp = fp[:-1]
                self.val_fps.append(os.path.join(self.naip_path, fp))

        self.naip_chips = glob.glob(self.naip_path + '/**/*.png', recursive=True)
        print("self.naip chips:", len(self.naip_chips))
        print("held out set:", len(self.val_fps))

        # Conditioning on S2.
        if datatype == 's2' or datatype == 's2_and_downsampled_naip' or datatype == 'just-s2':

            self.datapoints = []
            for n in self.naip_chips:

		# If this is the train dataset, ignore the subset of images that we want to use for validation.
                if self.split == 'train' and specify_val and (n in self.val_fps):
                    print("split == train and n in val_fps")
                    continue
		# If this is the validation dataset, ignore any images that aren't in the subset.
                if self.split == 'val' and specify_val and not (n in self.val_fps):
                    continue

                # ex. /data/first_ten_million/naip/m_2808033_sw_17_060_20191202/tci/36046_54754.png
                split_path = n.split('/')
                chip = split_path[-1][:-4]
                tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16
                s2_left_corner = tile[0] * 16, tile[1] * 16
                diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]

                s2_path = os.path.join(self.s2_path, str(tile[0])+'_'+str(tile[1]), str(diffs[1])+'_'+str(diffs[0])+'.png')

                self.datapoints.append([n, s2_path])

                # Only add 'max_tiles' datapoints to the datapoints list if specified.
                if self.max_tiles != -1 and len(self.datapoints) >= self.max_tiles:
                    break

            self.data_len = len(self.datapoints)

        # NAIP reconstruction, build downsampled version on-the-fly.
        elif datatype == 'naip':

            # Build list of NAIP chip paths.
            self.datapoints = []
            for n in self.naip_chips:

                # If this is the train dataset, ignore the subset of images that we want to use for validation.
                if self.split == 'train' and self.specify_val and (naip_path in self.val_fps):
                    continue
                # If this is the validation dataset, ignore any images that aren't in the subset.
                if self.split == 'val' and self.specify_val and not (naip_path in self.val_fps):
                    continue

                self.datapoints.append(n)
            self.data_len = len(self.datapoints)

        print("self.data_len:", self.data_len)

    def __getitem__(self, index):

        # Conditioning on S2, or S2 and downsampled NAIP.
        if self.datatype == 's2' or self.datatype == 's2_and_downsampled_naip' or self.datatype == 'just-s2':
            datapoint = self.datapoints[index]
            naip_path, s2_path = datapoint[0], datapoint[1]

            # Load the 512x512 NAIP chip.
            naip_chip = skimage.io.imread(naip_path)

            # Load the Tx32x32 S2 file.
            s2_images = skimage.io.imread(s2_path)
            s2_chunks = np.reshape(s2_images, (-1, 32, 32, 3))

            # SPECIAL CASE: when we are running a S2 upsampling experiment, sample 1 more 
            # S2 image than specified. We'll use that as our "high res" image and the rest 
            # as conditioning. Because the min number of S2 images is 18, have to use 17 for time series.
            if self.datatype == 'just-s2':
                rand_indices = random.sample(range(0, len(s2_chunks)), self.n_s2_images)
                s2_chunks = [s2_chunks[i] for i in rand_indices[1:]]
                s2_chunks = np.array(s2_chunks)
                naip_chip = s2_chunks[0]  # this is a fake naip chip
            else:
                # SPECIAL CASE: when we are running a S2 upsampling experiment, sample 1 more 
                # S2 image than specified. We'll use that as our "high res" image and the rest 
                # as conditioning. Because the min number of S2 images is 18, have to use 17 for time series.
                if self.datatype == 'just-s2':
                    rand_indices = random.sample(range(0, len(s2_chunks)), self.n_s2_images)
                    s2_chunks = [s2_chunks[i] for i in rand_indices[1:]]
                    s2_chunks = np.array(s2_chunks)
                    naip_chip = s2_chunks[0]  # this is a fake naip chip

                else:
                    # Iterate through the 32x32 chunks at each timestep, separating them into "good" (valid)
                    # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
                    goods, bads = [], []
                    for i,ts in enumerate(s2_chunks):
                        if [0, 0, 0] in ts:
                            bads.append(i)
                        else:
                            goods.append(i)

                    # Pick 18 random indices of s2 images to use. Skip ones that are partially black.
                    if len(goods) >= self.n_s2_images:
                        rand_indices = random.sample(goods, self.n_s2_images)
                    else:
                        need = self.n_s2_images - len(goods)
                        rand_indices = goods + random.sample(bads, need)

                    s2_chunks = [s2_chunks[i] for i in rand_indices]
                    s2_chunks = np.array(s2_chunks)

                # I *think* that the input should be the original 32x32 instead of upsampling first...
                # So deleted the code where we upsampled the S2 images to the desired output size.

            # If conditioning on downsampled naip (along with S2), need to downsample original NAIP datapoint and upsample
            # it to get it to the size of the other inputs.
            if self.datatype == 's2_and_downsampled_naip':
                downsampled_naip = cv2.resize(naip_chip, dsize=(self.downsample_res,self.downsample_res), interpolation=cv2.INTER_CUBIC)
                downsampled_naip = cv2.resize(downsampled_naip, dsize=(self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)

                if len(s2_chunks) == 1:
                    s2_chunk = s2_chunks[0]

                    [s2_chunk, downsampled_naip, naip_chip] = Util.transform_augment(
                                                   [s2_chunk, downsampled_naip, naip_chip], split=self.split, min_max=(-1, 1))
                    img_SR = torch.cat((s2_chunk, downsampled_naip))
                    img_HR = naip_chip
                else:
                    print("TO BE IMPLEMENTED")
                    [s2_chunks, downsampled_naip, naip_chip] = Util.transform_augment(
                                                    [s2_chunks, downsampled_naip, naip_chip], split=self.split, min_max=(-1, 1), multi_s2=True)
                    img_SR = torch.cat((torch.stack(s2_chunks), downsampled_naip))
                    img_HR = naip_chip

            elif self.datatype == 's2' or self.datatype == 'just-s2':

                if len(s2_chunks) == 1:
                    s2_chunk = s2_chunks[0]

                    [img_SR, img_HR] = Util.transform_augment(
				    [s2_chunk, naip_chip], split=self.split, min_max=(-1, 1))
                else:
                    # NOTE: trying something different here....
                    s2_chunks = [totensor(img) for img in s2_chunks]
                    img_HR = totensor(naip_chip)

                    #[s2_chunks, img_HR] = Util.transform_augment(
                    #                [s2_chunks, naip_chip], split=self.split, min_max=(-1, 1), multi_s2=True)

                    use_3d = False
                    if use_3d:
                        img_SR = torch.stack(s2_chunks)
                    else:
                        img_SR = torch.cat(s2_chunks)

            return {'gt': img_HR, 'lq': img_SR, 'Index': index}

        elif self.datatype == 'naip':
            naip_path = self.datapoints[index]

            # Load the 512x512 NAIP chip.
            naip_chip = skimage.io.imread(naip_path)

            # Create the downsampled version on-the-fly.
            downsampled_naip = cv2.resize(naip_chip, dsize=(self.downsample_res,self.downsample_res), interpolation=cv2.INTER_CUBIC)
            downsampled_naip = cv2.resize(downsampled_naip, dsize=(self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)

            [img_SR, img_HR] = Util.transform_augment([downsampled_naip, naip_chip], split=self.split, min_max=(-1, 1))

            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

        else:
            print("Unsupported type...")

    def __len__(self):
        return self.data_len
