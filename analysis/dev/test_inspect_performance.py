import pandas as pd


filepath = '/media/cp3fr/Elements/rss21_visual_attention/flightmare/analysis' \
           '/performance/gate-wall/performance.csv'

d = pd.read_csv(filepath)

ind = (
    (d.has_network_used.values == 1) &
    (d.has_dda.values == 1) &
    # (d.subject.values == 16) &
    # (d.run.values == 5) &
    # (d.li.values == 1) &
    (d.num_laps.values  > 1) &
    (d.has_ref.values == 1) &
    (d.has_state_q.values == 1) &
    (d.has_state_v.values == 1) &
    (d.has_state_w.values == 1) &
    (d.has_fts.values == 0) &
    (d.has_decfts.values == 0) &
    (d.has_attbr.values == 1) &
    (d.has_gztr.values == 0)
)


names = d.loc[ind, 'filepath'].values

count = 0
for n in names:
    print(count, n)
    count +=1

