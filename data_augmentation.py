import os
import pandas as pd
# import numpy as np
# import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from src.com.util.util_model import preprocess

# import nlpaug.augmenter.sentence as nas
# import nlpaug.flow as nafc

# from nlpaug.util import Action

aug = naw.ContextualWordEmbsAug(
    model_path='digitalepidemiologylab/covid-twitter-bert', action="substitute")


def augment_text(df):
    # aug_w2v.aug_p = pr
    # new_text = []

    # selecting the minority class samples
    df_n = df[df.label == 1].reset_index(drop=True)
    index = ['tweet_id', 'tweet', 'user_id', 'label']
    new_df = pd.DataFrame(columns=index)

    # data augmentation loop
    for i in (range(0, len(df_n))):
        print(i)
        text = df_n.iloc[i]['tweet']
        augmented_texts = aug.augment(preprocess(text), n=3)
        # print("************************************")
        # print(df_n.iloc[i]['tweet'])
        a = {
            "tweet_id": df_n.iloc[i]['tweet_id'],
            'user_id': df_n.iloc[i]['user_id'],
            "tweet": df_n.iloc[i]['tweet'],
            'label': df_n.iloc[i]['label']
        }
        series = pd.Series(a, index=index)
        new_df = new_df.append(series, ignore_index=True)
        for j in range(len(augmented_texts)):
            # new_df['tweet_id'] = int(df_n.iloc[i]['tweet_id']) + 1239172732690014208 + j
            # new_df['tweet'] = augmented_texts[j]
            # new_df['label'] = 1
            # new_df['user_id'] = df_n.iloc[i]['user_id']
            a = {
                "tweet_id": int(df_n.iloc[i]['tweet_id']) + 1239172732690014208 + j,
                'user_id': df_n.iloc[i]['user_id'],
                "tweet": augmented_texts[j],
                'label': 1
            }
            series = pd.Series(a, index=index)
            new_df = new_df.append(series, ignore_index=True)
            # print(augmented_texts[j])
            # new_text.append(augmented_text)
        # augmented_text = aug_w2v.augment(text)

    # dataframe
    # new = pd.DataFrame({'tweet': new_text, 'label': 1})
    # df = (df.append(new).reset_index(drop=True))
    return new_df


def data_augment():
    df = pd.read_csv('../../../data/train.tsv', sep='\t')
    df.columns = ['tweet_id', 'user_id', 'tweet', 'label']
    train = df[df.label == 1]
    train_aug = augment_text(train)
    train_aug = train_aug.append(df[df.label == 0], ignore_index=True)
    train_aug.to_csv(os.path.join('../../../data/train_aug3.tsv'), index=True, sep='\t')


if __name__ == "__main__":
    # data_augment()
    # # read data
    # df = pd.read_csv('../../../data/train.tsv', sep='\t')
    # df.columns = ['tweet_id', 'user_id', 'tweet', 'label']
    # df = df.reset_index(drop=True)

    aug_df = pd.read_csv('../../../data/train_aug3.tsv', sep='\t')
    # aug_df = aug_df.reset_index(drop=True)
    # aug_df1 = aug_df[aug_df.label == 1]
    # aug_df0 = aug_df[aug_df.label == 0]
    # print(len(aug_df))
    # aug_df.drop()
    aug_df = aug_df.reset_index(drop=True)
    # aug_df = aug_df.drop(['Unnamed: 0'], axis=1)
    # # print(aug_df.keys())
    aug_df.to_csv(os.path.join('../../../data/train_aug3.tsv'), index=False, header=1, sep='\t')
    # #
    # # # add data
    # # # remove nan
