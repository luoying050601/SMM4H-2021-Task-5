import os
import re
import html
import torch
import emoji
import numpy as np
# import regex
import unicodedata
import unidecode

control_char_regex = re.compile(r'[\r\n\t]+')
transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
LOG_DIR = Proj_dir + '/log'


def preprocess(text):
    # lower case
    text = text.lower()
    # replace USER TAG
    text = re.sub(r'(^|[^@\w])@(\w{1,15})\b', '@USER', text)
    # ^@[A-Za-z0-9_]{1,15}$
    # replace HTTP tag
    text = re.sub(r"http\S+", "URL", text)

    text = html.unescape(text)
    text = text.translate(transl_table)
    text = text.replace('…', '...')
    text = re.sub(control_char_regex, ' ', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = ' '.join(text.split())
    text = text.strip()
    # demojize
    text = emoji.demojize(text)
    text = unidecode.unidecode(text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')

    return text


# %%


class AverageMeter:
    """
    Computes and stores the average and current value
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_all(seed=42):
    """
  Fix seed for reproducibility
  """
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)


class Dataset:
    def __init__(self, text, label, config=None):
        self.text = text
        self.label = label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = process_data(
            self.text[item],
            self.tokenizer,
            self.max_len,
            # None
            self.label[item],
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'text': data['text'],
            'label': torch.tensor(int(data["label"]), dtype=torch.long),
        }


def process_data(org_text, tokenizer, max_len, label, add_special_tokens=True):
    text = preprocess(org_text)
    # text = org_text
    token_ids = tokenizer.encode(text, truncation=True, add_special_tokens=add_special_tokens, max_length=512)
    mask = [1] * len(token_ids)

    padding = max_len - len(token_ids)

    if padding >= 0:
        token_ids = token_ids + ([0] * padding)
        mask = mask + ([0] * padding)
    else:
        token_ids = token_ids[0:max_len]
        mask = mask[0:max_len]

    # label = 1 if label == 'INFORMATIVE' else 0

    assert len(token_ids) == max_len
    assert len(mask) == max_len

    return {'text': text,
            'ids': token_ids,
            'mask': mask,
            'label': label
            }


def make_print_to_file(path=LOG_DIR):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day and time:' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # print -> log
    #############################################################
    print(fileName.center(60, '*'))


class EarlyStopping:
    """
    Early stopping utility
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self, patience=7, mode="max", delta=0.001, bs=None):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.bs = bs

        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        # best_flag = False
        if self.mode == "min":
            score = -1.0 * epoch_score
            self.delta = -1.0 * self.delta
        else:
            score = np.copy(epoch_score)
        if self.best_score is None and self.bs is None:
            best_score = score
            print('current best score:', best_score)
            self.bs = score
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                self.counter = 0
        else:
            best_score = epoch_score
            print('current best score:', best_score)
            # best_flag = True
            self.bs = epoch_score
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0
        # return best_flag

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


if __name__ == "__main__":
    str1 = "We’re parking at the airport and my mom rolled down the window to speak to an attendant "
    str2 = "@Watchu28020892 @cmyeaton @ScottGottliebMD No testing therefore there are less infections.  "
    str3 = "I can’t with this family 🤧🤧 I literally said I might have Coronavirus "
    str4 = "so I've done a new page on how that affects stammering and the Equality Act. https://t.co/9RwzUxV4Kg"
    str5 = "My mom is in one of only 2 hospitals in the county I'm from &amp; NOBODY has even ASKED me."
    str6 = "There's an even deadlier virus than coronavirus, and it has murdered our " \
           "people for 500+ years... that virus is shyápu religion, "\
           "white"" religion."
    # str7 = "so I've done a new page on how that affects stammering and the Equality Act. https://t.co/9RwzUxV4Kg"
    # test base data
    # output = preprocess(str1)
    # print("1:" + str1 + "\n->\n" + output)
    # print('***************************************************')
    # # # test base data with @XXXX mark-> @USER
    # #
    # output = preprocess(str2)
    # print("2:" + str2 + "\n->\n" + output)
    # print('***************************************************')
    # # test base data with emoji-> [XXXX]
    # # DONE
    # output = preprocess(str3)
    # print("3:" + str3 + "\n->\n" + output)
    # print('***************************************************')
    # # test base data with  http-> HTTPURL
    # output = preprocess(str4)
    # print("4:" + str4 + "\n->\n" + output)
    # print('***************************************************')
    # # test base data with  http-> HTTPURL
    # output = preprocess(str5)
    # print("5:" + str5 + "\n->\n" + output)
    print('***************************************************')
    # test base data with  http-> HTTPURL
    output = preprocess(str6)
    print("6:" + str6 + "\n->\n" + output)
    print('***************************************************')
