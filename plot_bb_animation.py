import glob
import os
from videofig import videofig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.misc import imread

def run_sin():
  def redraw_fn(f, axes):
    amp = float(f) / 3000
    f0 = 3
    t = np.arange(0.0, 1.0, 0.001)
    s = amp * np.sin(2 * np.pi * f0 * t)
    if not redraw_fn.initialized:
      redraw_fn.l, = axes.plot(t, s, lw=2, color='red')
      redraw_fn.initialized = True
    else:
      redraw_fn.l.set_ydata(s)

  redraw_fn.initialized = False

  videofig(100, redraw_fn)    

def plot_img(arr, fs=(6,6), title=None):
  plt.figure(figsize=fs)
  plt.imshow(arr.astype('uint8'))
  plt.title(title)
  plt.show()
    
def load_img(fpath):
  img = cv2.imread(fpath)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def run_images(fpaths):
  fpaths.sort()
  def redraw_fn(f, axes):
    img_file = fpaths[f]
    img = imread(img_file)
    if not redraw_fn.initialized:
      redraw_fn.im = axes.imshow(img, animated=True)
      redraw_fn.initialized = True
    else:
      redraw_fn.im.set_array(img)
  redraw_fn.initialized = False
  videofig(len(fpaths), redraw_fn, play_fps=60)

def get_metadata_df(fpath):
  metadata = pd.read_csv(metadata_fpath)
  metadata['label_name'] = 'ball'
  metadata['label_id'] = 1
  fnames = metadata['filename']
  fpaths = [os.path.join(IMG_DIR, f) for f in fnames]
  metadata['fpath'] = fpaths
  metadata['fname'] = fnames
  return metadata


DATA_DIR = '/Users/bfortuner/bigguy/volleyball/'
IMG_DIR = os.path.join(DATA_DIR, 'images_subset')
metadata_fpath = os.path.join(IMG_DIR, 'bbox_labels.csv')

if __name__ == "__main__":
  meta = get_metadata_df(metadata_fpath)
  run_images(list(meta['fpath'].values))
