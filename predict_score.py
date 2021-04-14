import os
import time
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, BertForSequenceClassification, \
    RobertaTokenizer
# import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.utils.extmath import softmax
# from sklearn.metrics import classification_report, f1_score
from src.com.util.util_model import make_print_to_file, Dataset

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
LOG_DIR = Proj_dir + '/log'


class config:
    LR_LIST = [1e-5]
    # lr = 2e-5
    # SEED = random.randint(3, 5)  # 3 - 5
    KFOLD = 5
    SAVE_DIR = Proj_dir + '/post_evaluation/'
    LOAD_DIR = Proj_dir + '/run_predict/'
    TRAIN_FILE = '../../../data/train.tsv'
    VAL_FILE = '../../../data/valid.tsv'
    TEST_FILE = VAL_FILE

    # TEST_FILE = '../../../data/test.tsv'
    OOF_FILE = os.path.join(SAVE_DIR, 'output/oof.csv')
    MAX_LEN = 96
    MODEL = 'digitalepidemiologylab/covid-twitter-bert'
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
    EPOCHS_LIST = [10]
    BATCH_SIZE_LIST_T = [32]
    BATCH_SIZE_LIST_V = [32]
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def test_fn(data_loader, model, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    test_preds = []

    for bi, d in enumerate(tk0):
        ids = d['ids']
        mask = d['mask']
        # label = d['label']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        # label = label.to(device, dtype=torch.long)

        with torch.no_grad():
            k = model(input_ids=ids, attention_mask=mask)
            # loss = k['loss']
            logits = k['logits']

        logits = logits.detach().cpu().numpy()
        preds = softmax(logits)[:, 1]
        test_preds = test_preds + preds.tolist()

    return test_preds


def test_prediction_mutli():
    # print(config.SEED)
    df = pd.read_csv(config.TEST_FILE, sep='\t')
    # df.columns = ['tweet_id', 'user_id', 'tweet']
    df['label'] = 0
    # test_dataset = Dataset(
    #     text=df.tweet.values,
    #     label=df.label.values,
    #     config=config
    # )
    #
    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=config.BATCH_SIZE_LIST_T[0],
    #     num_workers=4
    # )

    scores = pd.DataFrame()
    scores['tweet_id'] = df['tweet_id']

    ensemble_list = [
        'model_rt_0_7426', 'model_rt_0_7426', 'model_rt_2_7426', 'model_rt_3_7426', 'model_rt_4_7426',
        'model_roberta_0_2078', 'model_roberta_1_2078', 'model_roberta_2_2078', 'model_roberta_3_2078',
        'model_roberta_4_2078',
        'model_0_76', 'model_1_76', 'model_2_76', 'model_3_76', 'model_4_76',
    ]
    # ensemble_list = [
    #     'm', 'model_rt_0_7426', 'model_rt_2_7426', 'model_rt_3_7426', 'model_rt_4_7426',
    #     'model_roberta_0_2078', 'model_roberta_1_2078', 'model_roberta_2_2078', 'model_roberta_3_2078',
    #     'model_roberta_4_2078',
    #     'model_0_76', 'model_1_76', 'model_2_76', 'model_3_76', 'model_4_76',
    # ]

    # ensemble_list = ['model_0_42', 'model_1_42', 'model_2_42', 'model_3_42', 'model_4_42',
    #                  'model_99_0', 'model_99_1', 'model_99_2', 'model_99_3', 'model_99_4',
    #                  #
    #                  'model_0_75', 'model_1_75', 'model_2_75', 'model_3_75', 'model_4_75',
    #                  # 'model_0_99', 'model_1_99', 'model_2_99', 'model_3_99', 'model_4_99'
    #                  ]
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = BertForSequenceClassification.from_pretrained(config.MODEL, num_labels=2)
    model.to(device)
    for i in ensemble_list:
        print(i)
        if i.__contains__('roberta'):
            MODEL = 'roberta-large'
            model = RobertaForSequenceClassification.from_pretrained(MODEL, num_labels=2)
            config.TOKENIZER = RobertaTokenizer.from_pretrained(MODEL)

        elif i.__contains__('rt'):
            MODEL = 'cardiffnlp/twitter-roberta-base'
            model = RobertaForSequenceClassification.from_pretrained(MODEL, num_labels=2)
            config.TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
        else:
            MODEL = 'digitalepidemiologylab/covid-twitter-bert'
            model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=2)
            config.TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
        model.to(device)
        test_dataset = Dataset(
            text=df.tweet.values,
            label=df.label.values,
            config=config
        )

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE_LIST_T[0],
            num_workers=4
        )
        model.load_state_dict(
            torch.load(os.path.join(config.LOAD_DIR, f'baseline/{i}.bin')))
        y_preds = test_fn(test_data_loader, model, device)
        scores[f'{i}'] = y_preds
        # print(i, y_preds)
    scores['avg'] = sum((scores[f'{i}'] for i in ensemble_list)) / len(ensemble_list)

    scores['average_prob'] = (scores['avg'] >= 0.5) * 1
    scores['label'] = scores['average_prob']

    scores.to_csv(os.path.join(config.SAVE_DIR, 'scores_val_all_15.csv'), index=False)
    # submission = pd.DataFrame(scores, columns=['tweet_id', 'label'])
    # # print(len(submission[submission.label == 1]))
    # # print(len(submission[submission.label == 0]))
    # submission.to_csv(os.path.join(config.SAVE_DIR, 'submission_1.tsv'), index=False, sep='\t')


def test_prediction_CT_SEED():
    # print(config.SEED)
    df = pd.read_csv(config.TEST_FILE, sep='\t')
    # df.columns = ['tweet_id', 'user_id', 'tweet']
    df['label'] = 0
    test_dataset = Dataset(
        text=df.tweet.values,
        label=df.label.values,
        config=config
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_LIST_T[0],
        num_workers=4
    )

    scores = pd.DataFrame()
    scores['tweet_id'] = df['tweet_id']

    ensemble_list = [
        'model_0_42', 'model_1_42', 'model_2_42', 'model_3_42', 'model_4_42',
        'model_0_75', 'model_1_75', 'model_2_75', 'model_3_75', 'model_4_75',
        'model_0_99', 'model_1_99', 'model_2_99', 'model_3_99', 'model_4_99'
    ]
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = BertForSequenceClassification.from_pretrained(config.MODEL, num_labels=2)
    model.to(device)
    for i in ensemble_list:
        print(i)

        model.load_state_dict(
            torch.load(os.path.join(config.LOAD_DIR, f'baseline/{i}.bin')))
        y_preds = test_fn(test_data_loader, model, device)
        scores[f'{i}'] = y_preds
        # print(i, y_preds)
    scores['avg'] = sum((scores[f'{i}'] for i in ensemble_list)) / len(ensemble_list)
    scores['average_prob'] = (scores['avg'] >= 0.5) * 1
    scores['label'] = scores['average_prob']

    scores.to_csv(config.SAVE_DIR+f'/scores_val_ct_15.csv', index=False)
    # submission = pd.DataFrame(scores, columns=['tweet_id', 'label'])
    # submission.to_csv(os.path.join(config.SAVE_DIR, '/submission_2.tsv'), index=False, sep='\t')


if __name__ == "__main__":
    start = time.perf_counter()
    make_print_to_file(path='.')
    # run_result()
    test_prediction_CT_SEED()
    test_prediction_mutli()
    end = time.perf_counter()
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)
