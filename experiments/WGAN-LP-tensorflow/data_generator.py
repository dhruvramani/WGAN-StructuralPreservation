# Modified from https://github.com/igul222/improved_wgan_training/blob/master/gan_toy.py

import numpy as np
import sklearn.datasets
import random

# This module referes 'Python Generator', not 'Generative Model'.


class GeneratorGaussians8(object):
    def __init__(self,
                 batch_size: int=256,
                 scale: float=2.,
                 center_coor_min: float=-1.,
                 center_coor_max: float=1.,
                 stdev: float=1.414):
        self.batch_size = batch_size
        self.stdev = stdev
        scale = scale
        diag_len = np.sqrt(center_coor_min**2 + center_coor_max**2)
        centers = [
            (center_coor_max, 0.),
            (center_coor_min, 0.),
            (0., center_coor_max),
            (0., center_coor_min),
            (center_coor_max / diag_len, center_coor_max / diag_len),
            (center_coor_max / diag_len, center_coor_min / diag_len),
            (center_coor_min / diag_len, center_coor_max / diag_len),
            (center_coor_min / diag_len, center_coor_min / diag_len)
        ]
        self.centers = [(scale * x, scale * y) for x, y in centers]

    def __iter__(self):
        while True:
            dataset = []
            for i in range(self.batch_size):
                point = np.random.randn(2) * .02
                center = random.choice(self.centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= self.stdev
            yield dataset


class Feed(object):
    '''Feed image data to training process. '''
    def __init__(self, data_directory, batch_size, ncached_batches=100, shuffle=False):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.batch_idx = 0
        # number of batches to preload into memory
        self.ncached_batches = ncached_batches

        # filenames for all files in data dir
        self.filenames = sorted([f for f in os.listdir(self.data_directory) \
            if os.path.isfile(os.path.join(self.data_directory, f))])

        if (shuffle):
            np.random.shuffle(self.filenames)

        # index of first batch preloaded in memory
        self.cached_batch_start = -sys.maxsize

    # figure out image shape from the first image
    def get_img_shape(self):
        path = os.path.join(self.data_directory, self.filenames[0])
        img = np.asarray(self.open_image(path))
        return (img.shape[0], img.shape[1])

    # convert from global batch index (ie. between 0 and total number of 
    # batches in the entire training set) to corresponding cached batch index (number between
    # 0 and number of batches worth that get cached)
    # Also loads more data if batch_idx is outide what is currently cached
    def cidx(self, batch_idx):
        # batch_idx outside range of cached batches?
        if (batch_idx < self.cached_batch_start or 
            batch_idx >= self.cached_batch_start + self.ncached_batches):

            # new cached_batch_start
            self.cached_batch_start = self.ncached_batches * \
                int(batch_idx / float(self.ncached_batches))
            # preload more batches
            self.load_cache(self.cached_batch_start)

        # index of batch in currently preloaded data
        return batch_idx % self.ncached_batches

    # number of batches in entire directory 
    def nbatches(self):
        return int(len(self.filenames) / float(self.batch_size))

    # loads and returns np array of image, converting grayscale images
    # to RGB if necessary
    def open_image(self, f):
        img = Image.open(f)
        array = np.asarray(img)
        if (len(array.shape) == 2): # only 2 dims, no color dim, so grayscale
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            array = np.asarray(rgbimg)
        return array


    # do the actual loading from disk   
    def load_cache(self, batch_idx):
        # end of cache
        end_batch_idx = min((batch_idx + self.ncached_batches), self.nbatches())

        start = batch_idx * self.batch_size
        end = end_batch_idx * self.batch_size

        # full paths
        cache_filepaths = [os.path.join(self.data_directory, f) for f in self.filenames[start:end]]

        self.imgs = np.asarray([self.open_image(f) for f in cache_filepaths])
        self.cached_batch_start = batch_idx

    # public method, returns the next batch_size worth of images
    '''
    def feed(self, batch_idx):
        cidx = self.cidx(batch_idx) 
        imgs = self.imgs[ cidx*self.batch_size:(cidx+1)*self.batch_size ]

        if (imgs.dtype == 'float64'):
            imgs = imgs.astype('float32')
            
        if (imgs.dtype == 'uint8'):
            imgs = imgs.astype('float32') / 255.0

        assert imgs.shape[0] > 0
        return imgs

    '''

    def __iter__(self):
        self.batch_idx = 0
        batches = self.nbatches()
        while(self.batch_idx < batches):
            cidx = self.cidx(self.batch_idx) 
            imgs = self.imgs[ cidx*self.batch_size:(cidx+1)*self.batch_size ]

            if (imgs.dtype == 'float64'):
                imgs = imgs.astype('float32')
            
            if (imgs.dtype == 'uint8'):
                imgs = imgs.astype('float32') / 255.0

            assert imgs.shape[0] > 0
            self.batch_idx += 1 
            yield imgs


class GeneratorGaussians25(object):
    def __init__(self,
                 batch_size: int=256,
                 n_init_loop: int=4000,
                 x_iter_range_min: int=-2,
                 x_iter_range_max: int=2,
                 y_iter_range_min: int=-2,
                 y_iter_range_max: int=2,
                 noise_const: float = 0.05,
                 stdev: float=2.828):
        self.batch_size = batch_size
        self.dataset = []
        for i in range(n_init_loop):
            for x in range(x_iter_range_min, x_iter_range_max+1):
                for y in range(y_iter_range_min, y_iter_range_max+1):
                    point = np.random.randn(2) * noise_const
                    point[0] += 2 * x
                    point[1] += 2 * y
                    self.dataset.append(point)
        self.dataset = np.array(self.dataset, dtype='float32')
        np.random.shuffle(self.dataset)
        self.dataset /= stdev

    def __iter__(self):
        while True:
            for i in range(int(len(self.dataset) / self.batch_size)):
                yield self.dataset[i * self.batch_size:(i + 1)*self.batch_size]


class GeneratorSwissRoll(object):
    def __init__(self,
                 batch_size: int=256,
                 noise_stdev: float=0.25,
                 stdev: float=7.5):
        self.batch_size = batch_size
        self.noise_stdev = noise_stdev
        self.stdev = stdev

    def __iter__(self):
        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=self.batch_size,
                noise=self.noise_stdev
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= self.stdev  # stdev plus a little
            yield data
