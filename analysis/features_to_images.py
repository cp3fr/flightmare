import numpy as np
from pyforest import *

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *
from PIL import Image
from pprint import pprint


def make_path(f):
    """Makes a filepath after checking if it already exists."""
    if isinstance(f,str):
        f=Path(f)
    if not f.exists():
        f.mkdir(parents=True, exist_ok=True)


def feature2image(x,xres,yres,minval=None,maxval=None):
    """Feature vector to PIL image."""
    if minval is None:
        minval=np.nanmin(x)
    if maxval is None:
        maxval=np.nanmax(x)
    m = np.zeros((xres, yres, 3), dtype=np.uint8)
    m[:,:,0]=(x.reshape(yres,xres).T-minval)/(maxval-minval)*255
    m[:,:,1]=m[:,:,0]
    m[:,:,2]=m[:,:,0]
    return Image.fromarray(m, 'RGB')


def scale_image(im,factor):
    """Scales a PIL image."""
    return im.resize((int(im.size[0]*factor),int(im.size[1]*factor)),
                     resample=Image.BOX)


xres=25
yres=19
minval=0
maxval=None
scale=32

# Find filepaths to encoder features data "encfts".
filepaths=sorted((base_path/'analysis'/'logs'/'dda_encfts-coll-totmedtraj'
    ).rglob('*.npy'))
for f in filepaths:
    o=Path(f.as_posix().replace('/logs/','/plot/features/').replace('.npy',
                                                                    '.jpg'))
    print('..processing {}'.format(o))
    if not o.exists():
        make_path(o.parent)
        x=np.load(f)
        im=feature2image(x,xres,yres,minval,maxval)
        im=scale_image(im,scale)
        im.save(o,'JPEG',quality=100,optimize=True,progressive=True)

# Load feature track data and save as csv files
# todo: convert feature track data to images
filepaths = sorted((base_path / 'analysis' / 'logs' /
    'dda_ftstr-coll-totmedtraj').rglob('*.npy'))
for f in filepaths:
    o = Path(f.as_posix().replace('/logs/', '/plot/features/').replace(
        '.npy','.csv'))
    print('..processing {}'.format(o))
    if not o.exists():
        make_path(o.parent)
        x=np.load(f,allow_pickle=True)
        x=x.item()
        df=pd.DataFrame(x).T.reset_index()
        # todo: figure out the meaning of each column variable.
        df.columns=['id','px','py','vx','vy','unknown']
        df['id']=df['id'].astype(int)
        df['f']=int(o.stem)
        df.to_csv(o,index=False)
        print(df)