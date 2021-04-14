import os
import time
import random
import torch
from transformers import AdamW, RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.utils.extmath import softmax
from sklearn import model_selection
from sklearn.metrics import classification_report, f1_score
from src.com.util.util_model import EarlyStopping, make_print_to_file, AverageMeter, Dataset, seed_all

# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from pathlib import Path
# import matplotlib.cm as cm
# from typing import *
# import emoji
# import regex
# import html
# import unicodedata
# import unidecode

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
# print(Proj_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

best_score = None


# %%


class config:
    # lr = 2e-5
    SEED = 2078
    # random.randint(0,10000)  # 3 - 5
    # random.randint(3, 5)  # 3-5
    KFOLD = 5
    SAVE_DIR = Proj_dir + '/run_roberta_large_oof'
    TRAIN_FILE = '../../../data/train.tsv'
    VAL_FILE = '../../../data/valid.tsv'
    # TEST_FILE = '../../../data/test.tsv'
    TEST_FILE = VAL_FILE
    OOF_FILE = os.path.join(SAVE_DIR, 'output/oof.csv')
    MAX_LEN = 96
    MODEL = 'roberta-large'
    TOKENIZER = RobertaTokenizer.from_pretrained(MODEL)
    # EPOCHS = 5
    EPOCHS_LIST = [10]
    LR_LIST = [1e-5]
    BATCH_SIZE_LIST_T = [32]
    # BATCH_SIZE_LIST_V = [16, 32]
    # TRAIN_BATCH_SIZE = 16
    # VALID_BATCH_SIZE = 16
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# %%

def train_fn(data_loader, model, optimizer, device):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        mask = d['mask']
        label = d['label']
        ids = ids.to(device, dtype=torch.long)
        label = label.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        model.zero_grad()
        k = model(input_ids=ids, labels=label, attention_mask=mask)
        loss = k['loss']
        # logits = k['logits']

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


# %%

def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    yt, yp = [], []

    for bi, d in enumerate(tk0):
        ids = d['ids']
        mask = d['mask']
        label = d['label']

        ids = ids.to(device, dtype=torch.long)
        label = label.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        with torch.no_grad():
            k = model(input_ids=ids, attention_mask=mask, labels=label)
            loss = k['loss']
            logits = k['logits']
            # loss, logits = model(input_ids=ids, attention_mask=mask, labels=label)

        logits = logits.detach().cpu().numpy()

        preds = softmax(logits)
        pred_labels = np.argmax(preds, axis=1).flatten()
        ground_labels = label.to('cpu').numpy()
        # print("predict label:", pred_labels.tolist(), ";actual label:", ground_labels.tolist())
        yt = yt + ground_labels.tolist()
        yp = yp + pred_labels.tolist()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)

    return f1_score(yt, yp)


# %%

def test_fn(data_loader, model, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    test_preds = []

    for bi, d in enumerate(tk0):
        ids = d['ids']
        mask = d['mask']
        label = d['label']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        label = label.to(device, dtype=torch.long)

        with torch.no_grad():
            k = model(input_ids=ids, attention_mask=mask, labels=label)
            # loss = k['loss']
            logits = k['logits']

        logits = logits.detach().cpu().numpy()
        preds = softmax(logits)[:, 1]
        test_preds = test_preds + preds.tolist()

    return test_preds


def run(df_train, df_val, fold=None):
    train_dataset = Dataset(
        text=df_train.tweet.values,
        label=df_train.label.values,
        config=config
    )

    valid_dataset = Dataset(
        text=df_val.tweet.values,
        label=df_val.label.values,
        config=config
    )
    es = EarlyStopping(patience=5, mode="max", bs=best_score)
    model = RobertaForSequenceClassification.from_pretrained(config.MODEL, num_labels=2)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    print('Starting training....')
    test_predictions_dic = {}
    for lr in config.LR_LIST:
        for batch_size_t in config.BATCH_SIZE_LIST_T:
            es.counter = 0
            es.early_stop = False
            for epoches in config.EPOCHS_LIST:
                train_data_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size_t,
                    num_workers=4
                )

                valid_data_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=batch_size_t,
                    num_workers=2
                )

                optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
                for epoch in range(epoches):
                    print(f'training epoch size= {epoches} , lr= {lr} ,batch size={batch_size_t} | Epoch :{epoch + 1}')
                    train_fn(train_data_loader, model, optimizer, device)
                    print(
                        f'validating epoch size= {epoches} , lr= {lr} ,batch size={batch_size_t} | Epoch :{epoch + 1}')
                    valid_loss = eval_fn(valid_data_loader, model, device)
                    print(f'epoch size= {epoches} , lr= {lr} ,batch size={batch_size_t} | Epoch :{epoch + 1} '
                          f'| Validation Score :{valid_loss}')
                    if fold is None:
                        es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR, f'model/model_roberta_' + str(
                            config.SEED) + '.bin'))
                    else:
                        es(valid_loss, model,
                           model_path=os.path.join(config.SAVE_DIR,
                                                   f'model/model_roberta_{fold}_' + str(config.SEED) + '.bin'))
                    if es.early_stop:
                        print('Early stopping')
                        es.early_stop = False
                        # es.counter = 0
                        break

                print('Predicting for OOF')
                if fold is None:
                    model.load_state_dict(
                        torch.load(os.path.join(config.SAVE_DIR, 'model/model_roberta_' + str(config.SEED) + '.bin')))
                else:
                    model.load_state_dict(torch.load(
                        os.path.join(config.SAVE_DIR, f'model/model_roberta_{fold}_' + str(config.SEED) + '.bin')))

                model.to(device)

                test_predictions = test_fn(valid_data_loader, model, device)
                print(
                    'acc={} in epoches={} batch_size={} lr={}'.format(np.mean(test_predictions), epoches, batch_size_t,
                                                                      lr))
                test_predictions_dic[str(epoches) + '-' + str(batch_size_t) + '-' + str(lr)] = test_predictions

    return test_predictions, test_predictions_dic


# %%


def run_fold(fold_idx):
    """
      Perform k-fold cross-validation
    """
    seed_all(seed=config.SEED)
    df_train = pd.read_csv(config.TRAIN_FILE, sep='\t')
    df_train.columns = ['tweet_id', 'user_id', 'tweet', 'label']
    # df_val = pd.read_csv(config.VAL_FILE, sep='\t')
    # print(df_val.columns)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    train = df_train

    # concatenating train and validation set
    # train = pd.concat([df_train, df_val]).reset_index()

    # dividing folds
    kf = model_selection.StratifiedKFold(n_splits=config.KFOLD, shuffle=False, random_state=config.SEED)
    idx = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=train, y=train.label.values)):
        train.loc[val_idx, 'kfold'] = int(fold)
        if fold == fold_idx:
            idx = val_idx

    if os.path.isfile(config.OOF_FILE):
        scores = pd.read_csv(config.OOF_FILE)
        print('Found oof file')
    else:
        scores = train.copy()
        scores['oof'] = 0
        scores.to_csv(config.OOF_FILE, index=False)
        print('Created oof file')

    df_train = train[train.kfold != fold_idx]
    df_val = train[train.kfold == fold_idx]
    # max_y = []

    y, y_dic = run(df_train, df_val, fold_idx)
    # if max_y < acc:
    #     max_y = acc
    scores.loc[idx, 'oof'] = y

    scores.to_csv(config.OOF_FILE, index=False)


def test_prediction(kfold=None):
    # print(config.SEED)
    df = pd.read_csv(config.TEST_FILE, sep='\t')
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

    model = RobertaForSequenceClassification.from_pretrained(config.MODEL, num_labels=2)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model.to(device)

    if kfold is not None:
        for i in range(config.KFOLD):
            model.load_state_dict(
                torch.load(os.path.join(config.SAVE_DIR, f'model/model_roberta_{i}_' + str(config.SEED) + '.bin')))
            y_preds = test_fn(test_data_loader, model, device)
            scores[f'prob_{i}'] = y_preds
        scores['avg'] = (scores['prob_0'] + scores['prob_1'] + scores['prob_2'] + scores['prob_3'] + scores[
            'prob_4']) / 5
        scores['vote_1'] = (scores['prob_0'] >= 0.5) * 1 + (scores['prob_1'] >= 0.5) * 1 + (scores['prob_2'] >= 0.5) * 1 \
                           + (scores['prob_3'] >= 0.5) * 1 + (scores['prob_4'] >= 0.5) * 1
        scores['vote_0'] = 5 - scores['vote_1']
        # max vote
        scores['max_vote'] = (scores['vote_1'] >= scores['vote_0']) * 1
        scores['max_prob'] = (np.max(
            [scores['prob_0'], scores['prob_1'], scores['prob_2'], scores['prob_3'], scores['prob_4']],
            axis=0) >= 0.5) * 1
        # average probality
        scores['average_prob'] = (scores['avg'] >= 0.5) * 1
        scores['label'] = scores['average_prob']

        # print(i, y_preds)
    else:
        model.load_state_dict(
            torch.load(os.path.join(config.SAVE_DIR, f'model/model_roberta_' + str(config.SEED) + '.bin')))
        y_preds = test_fn(test_data_loader, model, device)
        scores[f'prob'] = y_preds
        scores['preds'] = (scores['prob'] >= 0.5) * 1
        scores['label'] = scores['preds']

    print(len(scores[scores.average_prob == 0]))
    print(len(scores[scores.max_prob == 0]))
    print(len(scores[scores.max_vote == 0]))
    scores.to_csv(os.path.join(config.SAVE_DIR, 'output/scores.csv'), index=False)
    submission = pd.DataFrame(df, columns=['tweet_id', 'label'])
    submission.to_csv(os.path.join(config.SAVE_DIR, 'output/submission_final.tsv'), index=False)

    # with open(os.path.join(config.SAVE_DIR, 'submission.tsv'), 'w') as f:
    #     for i in scores['label'].values:
    #         f.write(i + '\n')


def run_result():
    df = pd.read_csv(config.OOF_FILE)
    df['gold'] = df['label']
    # .map({'INFORMATIVE': 1, 'UNINFORMATIVE': 0})
    df.head(3)
    df['pred'] = (df['oof'] >= 0.5) * 1
    print(classification_report(df['gold'].values, df['pred'].values))
    from sklearn.metrics import roc_auc_score

    roc_auc_score(df['gold'].values, df['oof'].values)
    thresholds = np.arange(0, 1, 0.001)
    fscores = [f1_score(df['gold'].values, (df['oof'] >= t) * 1) for t in thresholds]
    idx = np.argmax(fscores)
    print(thresholds[idx], fscores[idx])


if __name__ == "__main__":
    start = time.perf_counter()
    make_print_to_file(path='.')
    # df_train = pd.read_csv(config.TRAIN_FILE, header=None, sep='\t')
    # df_val = pd.read_csv(config.VAL_FILE, sep='\t')
    # df_train.columns = ['tweet_id', 'user_id', 'tweet', 'label']
    # run(df_train, df_val)
    print(config.SEED)
    # run_fold(0)
    # run_fold(1)
    # run_fold(2)
    # run_fold(3)
    # run_fold(4)
    # run_result()
    test_prediction(kfold=config.KFOLD)
    end = time.perf_counter()
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)
