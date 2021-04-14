import os
import pandas as pd
from sklearn.metrics import classification_report

OOF_FILE = '../../../post_evaluation/post_val_average.tsv' # 77.6
# OOF_FILE = '../../../post_evaluation/post_val_max_voting.tsv' # 76
# OOF_FILE = '../../../post_evaluation/post_1.tsv'
# OOF_FILE = '../../../post_evaluation/post_2.tsv'
# OOF_FILE = '../../../post_evaluation/post_3.tsv'


# OOF_FILE = '../../../data/comparasion_valid.tsv'


def run_result():
    df = pd.read_csv(OOF_FILE, delimiter="\t")
    df['gold'] = df['ptrue']
    df.head(3)
    df['pred'] = df['label']
    print(classification_report(df['gold'].values, df['pred'].values))


Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
# test_file = '../../../data/test.tsv'
test_file = '../../../data/valid.tsv'
# all_avg_file = '../../../post_evaluation/submission_1.tsv'
# ct_avg_file = '../../../post_evaluation/submission_2.tsv'
# ct_ensemble_file = '../../../post_evaluation/submission_3.tsv'
all_avg_file = '../../../post_evaluation/scores_val_all_15.csv'
ct_avg_file = '../../../post_evaluation/scores_val_ct_15.csv'
ct_ensemble_file = '../../../post_evaluation/scores_val_ct_3.csv'
all_ensemble_file = '../../../post_evaluation/scores_val_all_3.csv'

test = pd.read_csv(test_file, sep='\t')
ct_avg = pd.read_csv(ct_avg_file, sep=',')
ct_ensemble = pd.read_csv(ct_ensemble_file, sep=',')
all_avg = pd.read_csv(all_avg_file, sep=',')
all_ensemble = pd.read_csv(all_ensemble_file, sep=',')
# # final99 = pd.read_csv(final99_file, sep='\t')
index = ['tweet_id','ptrue', 'ct_avg', 'ct_ensemble', 'all_avg',
         'all_ensemble','average','vote_1','vote_0','label'
         ]
compare_df = pd.DataFrame(columns=index)
#
# compare_df['tweet_id'] = test['tweet_id']
for i in range(len(test)):
    print(i)
    a = {
        "tweet_id": str(test.loc[i]['tweet_id']),
        "ptrue":test.loc[i]['label'],
        # 1 77.6
        "ct_avg": ct_avg.iloc[i]['avg'],
        "ct_ensemble": ct_ensemble.iloc[i]['avg'],
        "all_avg": all_avg.iloc[i]['avg'],
        "all_ensemble": all_ensemble.loc[i]['label'],
    }

    series = pd.Series(a, index=index)
    compare_df = compare_df.append(series, ignore_index=True)
#
compare_df['vote_1'] = compare_df['ct_avg'] + compare_df['ct_ensemble'] + \
                     compare_df['all_avg']
compare_df['vote_0'] = 3 - compare_df['vote_1']
compare_df['label'] = (compare_df['vote_1'] > compare_df['vote_0']) * 1
# 1
# compare_df['average'] = (compare_df['ct_avg'] + compare_df['ct_ensemble'] + \
#                         compare_df['all_avg'])/3
# # 2
# compare_df['average'] = (compare_df['ct_avg'] + compare_df['all_ensemble'] + \
#                         compare_df['all_avg'])/3
# # 3
# compare_df['average'] = (compare_df['ct_ensemble'] + compare_df['all_ensemble'] + \
#                         compare_df['all_avg'])/3
#4
# compare_df['average'] = (compare_df['ct_avg'] + compare_df['all_ensemble'] + \
#                         compare_df['all_avg'])/3
# compare_df['label'] = (compare_df['average'] >= 0.5) * 1
print(len(compare_df))
compare_df.to_csv(OOF_FILE, columns=['tweet_id','ptrue','label'],sep='\t', index=False)

run_result()
