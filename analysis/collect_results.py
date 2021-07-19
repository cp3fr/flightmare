import os
import sys
from pathlib import Path

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *

# Some settings.
colliders = 'gate-wall'

# Process new logfiles.
outpath = base_path/'analysis'/'tmp'/'performance.csv'
if not outpath.exists():
    filepaths = sorted((base_path/'analysis'/'process').rglob(
        '*/features_{}.csv'.format(colliders)))
    data = pd.DataFrame([])
    for f in filepaths:
        print(f.parts)
        df = pd.read_csv(f)
        df['model'] = f.parts[-4].split('_')[1]
        if f.parts[-3].find('trajectory')>-1:
            df['track'] = f.parts[-3].split('_')[0].split('-')[1]
            df['subject'] = int(f.parts[-3].split('_')[0].split('-')[2].replace('s',''))
            df['dataset'] = f.parts[-3].split('_')[0].split('-')[3]
        else:
            df['track'] = f.parts[-3].split('_')[2]
            df['subject'] = int(f.parts[-3].split('_')[0].replace('s', ''))
            df['dataset'] = 'test'
        df['mt'] = int(f.parts[-2].split('_')[1].split('-')[1])
        df['st'] = int(f.parts[-2].split('_')[2].split('-')[1])
        df['trial'] = int(f.parts[-2].split('_')[3])
        data=data.append(df)
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(outpath, index=False)

# Extract performance from logfiles.
inpath = base_path/'analysis'/'tmp'/'performance.csv'
outpath = base_path/'analysis'/'tmp'/'summary.csv'
if (inpath.exists()) & (not outpath.exists()):
    data = pd.read_csv(inpath)
    ddict={}
    for model in data['model'].unique():
        for track in data['track'].unique():
            for dataset in data['dataset'].unique():
                print('--------------------------------')
                for subject in data['subject'].unique():
                    ind = (
                        (data['model'].values == model) &
                        (data['track'].values == track) &
                        (data['dataset'].values == dataset) &
                        (data['subject'].values == subject)
                        )
                    num_samples = np.sum(ind)
                    num_coll_free = np.sum(data.loc[ind,
                                                        'num_collisions'].values==0)
                    prop_coll_free = num_coll_free / num_samples
                    ddict.setdefault('model',[])
                    ddict['model'].append(model)
                    ddict.setdefault('track', [])
                    ddict['track'].append(track)
                    ddict.setdefault('dataset', [])
                    ddict['dataset'].append(dataset)
                    ddict.setdefault('subject', [])
                    ddict['subject'].append(subject)
                    ddict.setdefault('num_samples', [])
                    ddict['num_samples'].append(num_samples)
                    ddict.setdefault('num_coll_free', [])
                    ddict['num_coll_free'].append(num_coll_free)
                    ddict.setdefault('prop_coll_free', [])
                    ddict['prop_coll_free'].append(prop_coll_free)
                    print(model, track, dataset, subject, num_samples,
                          num_coll_free, prop_coll_free)
    summary = pd.DataFrame(ddict)
    summary.to_csv(outpath, index=False)

# Plot performance tables and figure.
inpath = base_path/'analysis'/'tmp'/'performance.csv'
inpath2 = base_path/'analysis'/'tmp'/'summary.csv'
outpath = base_path/'analysis'/'tmp'/'plots'
if inpath2.exists():
    # Load performance data
    data = pd.read_csv(inpath)
    summary = pd.read_csv(inpath2)
    # Loop over different model configurations
    for model in data['model'].unique():
        for track in data['track'].unique():
            for dataset in data['dataset'].unique():
                # Make a figure that shows trajectories for all subjects
                fig,axs=plt.subplots(5,4)
                fig.set_figwidth(18)
                fig.set_figheight(10)
                axs = axs.flatten()
                i=0
                for subject in data['subject'].unique():
                    # Determine Success rate
                    ind = ((summary['model'].values == model) &
                           (summary['track'].values == track) &
                           (summary['dataset'].values == dataset) &
                           (summary['subject'].values == subject))
                    is_success = True
                    fontweight = 'normal'
                    fontcolor = 'black'
                    frame_highlight = False
                    success_rate = ''
                    if np.sum(ind) > 0:
                        v = summary.loc[ind, 'prop_coll_free'].iloc[0]
                        if not np.isnan(v):
                            success_rate = ' | {:.0f}%'.format(v * 100)
                            if v < 1:
                                fontweight = 'bold'
                                frame_highlight = True
                                is_success = False
                                fontcolor='red'
                    # Plot trajectory
                    ind = (
                        (data['model'].values == model) &
                        (data['track'].values == track) &
                        (data['dataset'].values == dataset) &
                        (data['subject'].values == subject) &
                        (data['trial'].values == 0)
                        )
                    if np.sum(ind)>0:
                        f=(Path(data.loc[ind, 'filepath'].iloc[0])
                           .parent/'trajectory-with-gates_gate-wall_045x270.jpg')
                        im=cv2.imread(f.as_posix())
                        #crop image borders
                        im=im[270:-340, 250:-250, :]
                        #add color frame (if not full success)
                        if not is_success:
                            im = cv2.copyMakeBorder(im,20,20,20,20,
                                cv2.BORDER_CONSTANT,value=(255,0,0))
                        axs[i].imshow(im)
                    axs[i].axis('off')
                    axs[i].set_title('s%03d'%subject + success_rate,
                                     fontweight=fontweight, color=fontcolor)
                    #raise the panel counter
                    i+=1
                # remove axis from remaining panels
                for i in range(i,axs.shape[0]):
                    axs[i].axis('off')
                plt.tight_layout()
                # make output directory
                if not outpath.exists():
                    outpath.mkdir(parents=True,exist_ok=True)
                # save the figure
                op = (outpath/('trajectories_{}_{}_{}.jpg'.format(
                    model,track,dataset)))
                fig.savefig(op.as_posix())
                plt.close(fig)
                fig=None
                axs=None
                #Pring overall success to prompt
                ind = ((summary['model'].values == model) &
                       (summary['track'].values == track) &
                       (summary['dataset'].values == dataset))
                num_samples = np.nansum(summary.loc[ind, 'num_samples'].values)
                num_coll_free = np.nansum(summary.loc[ind,
                                                      'num_coll_free'].values)
                prop_coll_free = np.nan
                if num_samples > 0:
                    prop_coll_free = num_coll_free/num_samples
                print('Success trajectories: {}/{} [{:.0f}%]'.format(
                    num_coll_free,num_samples,100*prop_coll_free))