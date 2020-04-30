import time
import os
import numpy as np
import glob
import subprocess
import re
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from text_token import _UNK, _PAD, _BOS, _EOS
from sumeval.metrics.rouge import RougeCalculator
from sklearn.metrics import f1_score

rouge = RougeCalculator(stopwords=True, lang="en")
smoothing = SmoothingFunction()

# Checker
def check_dir(dir_path):
    '''
    ans = input(
            "Are you going to delete all old files in {} ? [y/n] (y) "
            .format(dir_path))
    if ans == 'n':
        exit()
    else:
    '''
    '''
    files = glob.glob(os.path.join(dir_path, "*"))
    for f in files:
        os.remove(f)
    '''
    pass

def check_usage():
    output = subprocess.check_output(
            'nvidia-smi --query-gpu=memory.free --format=csv | tail -1',
            shell=True)
    free_mem = int(output.decode("utf-8").split(' ')[0])
    unit = output.decode("utf-8").split(' ')[1].strip()
    print_time_info("Free memory: {} {}".format(free_mem, unit))


def handle_model_dirs(model_dir, log_dir, dir_name, replace_model, is_load):
    if not replace_model:
        model_dir = os.path.join(model_dir, dir_name)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    else:
        if not is_load:
            check_dir(model_dir)

    log_dir = os.path.join(log_dir, dir_name)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, "validation"))

    return model_dir, log_dir


# Time
def get_time():
    T = time.gmtime()
    Y, M, D = T.tm_year, T.tm_mon, T.tm_mday
    h, m, s = T.tm_hour, T.tm_min, T.tm_sec
    return Y, M, D, h, m, s


def print_time_info(string):
    Y, M, D, h, m, s = get_time()
    _string = re.sub('[ \n]+', ' ', string)
    print("[{}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}] {}".format(
        Y, M, D, h, m, s, _string))


def print_curriculum_status(layer):
    print("###################################")
    print("#    CURRICULUM STATUS: LAYER{}    #".format(layer))
    print("###################################")


# Metric helper
class SequenceScorer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.single_bleu_scores = []
        self.bleu_scores = []
        self.single_rouge_scores = []
        self.rouge_scores = []
        self.best_rouge_scores = []

    def update(self, labels, refs, decode_result):
        self.single_bleu_scores.extend(single_BLEU(labels, decode_result))
        self.bleu_scores.extend(BLEU(refs, decode_result))

        self.single_rouge_scores.extend(single_ROUGE(labels, decode_result))
        self.rouge_scores.extend(ROUGE(refs, decode_result))
        self.best_rouge_scores.extend(best_ROUGE(refs, decode_result))

    def get_avg_scores(self):
        avg_single_bleu = np.mean(self.single_bleu_scores)
        avg_bleu = np.mean(self.bleu_scores)

        avg_single_rouge = np.mean(self.single_rouge_scores, axis=0)
        avg_rouge = np.mean(self.rouge_scores, axis=0)
        avg_best_rouge = np.mean(self.best_rouge_scores, axis=0)

        return (avg_single_bleu, avg_bleu, avg_single_rouge,
                avg_rouge, avg_best_rouge)

    def print_avg_scores(self):
        avg_single_bleu, avg_bleu, avg_single_rouge, \
            avg_rouge, avg_best_rouge = self.get_avg_scores()
        print(f"Average BLEU (single reference): {avg_single_bleu}")
        print(f"Average BLEU (multiple references): {avg_bleu}")
        print(f"Average ROUGE (single reference): {avg_single_rouge}")
        print(f"Average ROUGE (multiple references): {avg_rouge}")
        print(f"Average Best ROUGE (multiple references): {avg_best_rouge}")

    def write_avg_scores_to_file(self, fileobj):
        avg_single_bleu, avg_bleu, avg_single_rouge, \
            avg_rouge, avg_best_rouge = self.get_avg_scores()
        fileobj.write("{}\n{}\n{}\n{}\n{}\n".format(
            avg_single_bleu, avg_bleu,
            ', '.join(map(str, avg_single_rouge)),
            ', '.join(map(str, avg_rouge)),
            ', '.join(map(str, avg_best_rouge))
        ))


class MultilabelScorer:
    def __init__(self, f1_per_sample=False):
        self.f1_per_sample = f1_per_sample
        self.clear()

    def clear(self):
        self.micro_f1 = []
        self.weighted_f1 = []
        # self.samples_f1 = []

    def update(self, labels, prediction):
        # micro_f1, weighted_f1, samples_f1 = F1(labels, prediction)
        micro_f1, weighted_f1 = F1(labels, prediction, self.f1_per_sample)
        if self.f1_per_sample:
            self.micro_f1.extend(micro_f1)
            self.weighted_f1.extend(weighted_f1)
        else:
            self.micro_f1.append(micro_f1)
            self.weighted_f1.append(weighted_f1)
        # self.samples_f1.extend(samples_f1)

    def get_avg_scores(self):
        avg_micro_f1 = np.mean(self.micro_f1)
        avg_weighted_f1 = np.mean(self.weighted_f1)
        # avg_samples_f1 = np.mean(self.samples_f1)

        return (
            avg_micro_f1,
            avg_weighted_f1
        )

    def print_avg_scores(self):
        # avg_micro_f1, avg_weighted_f1, avg_samples_f1 = self.get_avg_scores()
        avg_micro_f1, avg_weighted_f1 = self.get_avg_scores()
        print(f"Average micro f1: {avg_micro_f1}")
        print(f"Average weighted f1: {avg_weighted_f1}")
        # print(f"Average samples f1: {avg_samples_f1}")

    def write_avg_scores_to_file(self, fileobj):
        '''
        avg_micro_f1, avg_weighted_f1, avg_samples_f1 = self.get_avg_scores()
        fileobj.write(f"{avg_micro_f1}\n{avg_weighted_f1}\n{avg_samples_f1}\n")
        '''
        avg_micro_f1, avg_weighted_f1 = self.get_avg_scores()
        fileobj.write(f"{avg_micro_f1}\n{avg_weighted_f1}\n")


def subseq(seq):
    # get subseq before the first _EOS token
    index_list = np.where(seq == _EOS)[0]
    if len(index_list) > 0:
        return seq[:index_list[0]]
    else:
        return seq

def micro_f1(y_true, y_pred):
    tp = ((y_true==1) & (y_pred==1)).sum()
    fp = ((y_true==0) & (y_pred==1)).sum()
    fn = ((y_true==1) & (y_pred==0)).sum()
    if tp == 0:
        return 0.0
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    return 2 * (precision * recall) / (precision + recall)

# NOTE: both labels and prediction should be n-hot vectors
def F1(labels, prediction, per_sample=False):
    def F1_per_sample(labels, prediction, average):
        if average == 'micro':
            return [
                micro_f1(label, pred)
                for label, pred in zip(labels, prediction)
            ]
        else:
            return [
                f1_score(y_true=np.array([label]), y_pred=np.array([pred]), average=average)
                for label, pred in zip(labels, prediction)
            ]

    if per_sample:
        return (
            F1_per_sample(labels, prediction, average='micro'),
            F1_per_sample(labels, prediction, average='weighted')
        )
    else:
        return (
            f1_score(labels, prediction, average='micro'),
            f1_score(labels, prediction, average='weighted')
        )


def BLEU(labels, hypothesis):
    # calculate the average BLEU score, trim paddings in labels
    bleu_score = [nltk.translate.bleu_score.sentence_bleu(
            [list(filter(lambda x: x != _PAD and x != _EOS, subseq(r))) for r in refs],
            list(filter(lambda x: x != _PAD and x != _EOS, subseq(h))),
            smoothing_function=smoothing.method5
        )
        for refs, h in zip(labels, hypothesis)
    ]
    return bleu_score


def single_BLEU(labels, hypothesis):
    # calculate the average BLEU score, trim paddings in labels
    bleu_score = [nltk.translate.bleu_score.sentence_bleu(
            [list(filter(lambda x: x != _PAD and x != _EOS, subseq(r)))],
            list(filter(lambda x: x != _PAD and x != _EOS, subseq(h))),
            smoothing_function=smoothing.method5
        )
        for r, h in np.stack(
            (labels, hypothesis), axis=1)
    ]
    return bleu_score


def single_ROUGE_1(labels, hypothesis):
    # use the subseq before the first _EOS token
    rouge_score = [
        rouge.rouge_n(
            references=' '.join([str(i) for i in list(
                filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
            summary=' '.join([str(i) for i in list(
                filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
            n=1
        )
        for r, h in np.stack((labels, hypothesis), axis=1)
    ]

    return rouge_score


def single_ROUGE(labels, hypothesis):
    # [[rouge_1, rouge_2, rouge_l, rouge_be], ...]
    # use the subseq before the first _EOS token
    rouge_score = [
        [
            rouge.rouge_n(
                references=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
                n=1
            ),
            rouge.rouge_n(
                references=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
                n=2
            ),
            rouge.rouge_l(
                references=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))])
            )
        ]
        for r, h in np.stack(
            (labels, hypothesis), axis=1)
    ]

    return rouge_score


def ROUGE(labels, hypothesis):
    # [[rouge_1, rouge_2, rouge_l, rouge_be], ...]
    # use the subseq before the first _EOS token
    rouge_score = [
        [
            rouge.rouge_n(
                references=[' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))])
                    for r in refs
                ],
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
                n=1
            ),
            rouge.rouge_n(
                references=[' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))])
                    for r in refs
                ],
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
                n=2
            ),
            rouge.rouge_l(
                references=[' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(r)))])
                    for r in refs
                ],
                summary=' '.join([str(i) for i in list(
                    filter(lambda x: x != _PAD and x != _EOS, subseq(h)))])
            )
        ]
        for refs, h in zip(labels, hypothesis)
    ]

    return rouge_score


def best_ROUGE(labels, hypothesis):
    # [[rouge_1, rouge_2, rouge_l, rouge_be], ...]
    # use the subseq before the first _EOS token
    rouge_score = [
        np.max(
            [
                [
                    rouge.rouge_n(
                        references=' '.join([str(i) for i in list(
                            filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                        summary=' '.join([str(i) for i in list(
                            filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
                        n=1
                    ),
                    rouge.rouge_n(
                        references=' '.join([str(i) for i in list(
                            filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                        summary=' '.join([str(i) for i in list(
                            filter(lambda x: x != _PAD and x != _EOS, subseq(h)))]),
                        n=2
                    ),
                    rouge.rouge_l(
                        references=' '.join([str(i) for i in list(
                            filter(lambda x: x != _PAD and x != _EOS, subseq(r)))]),
                        summary=' '.join([str(i) for i in list(
                            filter(lambda x: x != _PAD and x != _EOS, subseq(h)))])
                    )
                ]
                for r in refs
            ], axis=0)
        for refs, h in zip(labels, hypothesis)
    ]

    return rouge_score


# Argument helper
def add_path(args):
    args.embeddings_dir = os.path.join(args.data_dir, "GloVe")
    args.model_dir = os.path.join(args.data_dir, "model_slt")
    args.log_dir = os.path.join(args.data_dir, "log_slt")
    args.train_data_file = os.path.join(
            args.data_dir, "{}_train_data.pkl".format(args.dataset))
    args.valid_data_file = os.path.join(
            args.data_dir, "{}_valid_data.pkl".format(args.dataset))
    args.vocab_file = os.path.join(
            args.data_dir, "{}_vocab.pkl".format(args.dataset))
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    return args


def print_config(args):
    print()
    print("{}:".format("PATH"))
    print("\t{}: {}".format("Data directory", args.data_dir))
    print("\t{}: {}".format("Embeddings directory", args.embeddings_dir))
    print("\t{}: {}".format("Model directory", args.model_dir))
    print("\t{}: {}".format("Log directory", args.log_dir))
    print("\t{}: {}".format("Processed train data file", args.train_data_file))
    print("\t{}: {}".format("Processed valid data file", args.valid_data_file))
    print("\t{}: {}".format("Processed vocab file", args.vocab_file))
    print("{}:".format("DATA"))
    print("\t{}: {}".format("Dataset", args.dataset))
    print("\t{}: {}".format("Regenerate dataset", bool(args.regen)))
    if args.verbose_level > 1:
        print("\t{}:".format("Global parsing config"))
        print("\t\t{}: {}".format(
            "Use SpaCy or NLTK", "SpaCy" if args.is_spacy else "NLTK"))
        print("\t\t{}: {}".format(
            "Lemmatize the verbs or not", "Y" if args.is_lemma else "N"))
        print("\t\t{}: {}".format(
            "Using punctuation or not", "Y" if args.use_punct else "N"))
    if args.dataset == "E2ENLG":
        print("\t{}: {}".format("Fold attributes", bool(args.fold_attr)))
    print("\t{}: {}".format("Vocab size", args.vocab_size))
    print("\t{}: {}".format(
        "Use pretrained embeddings", bool(args.use_embedding)))
    if args.verbose_level > 1:
        print("\t{}: {}".format(
            "Encoder input max length", "No limit"
            if args.en_max_length == -1 else args.en_max_length))
        print("\t{}: {}".format(
            "Decoder input max length", "No limit"
            if args.de_max_length == -1 else args.de_max_length))
        print("\t{}: {}".format(
            "Decoder output min length", "No limit"
            if args.min_length == -1 else args.min_length))
    print("{}:".format("MODEL"))
    print("\t{}: {}".format("RNN cell", args.cell))
    print("\t{}: {}".format("Decoder layers", args.n_layers))
    print("\t{}: {}".format("RNN layers of encoder", args.n_en_layers))
    print("\t{}: {}".format("RNN layers of decoder", args.n_de_layers))
    print("\t{}: {}".format("Hidden size of encoder RNN", args.en_hidden_size))
    print("\t{}: {}".format("Hidden size of decoder RNN", args.de_hidden_size))
    print("\t{}: {}".format(
        "Encoder embedding layer", bool(args.en_embedding)))
    if args.en_embedding:
        print("\t{}: {}".format(
            "Shared embedding layer", bool(args.share_embedding)))
    if args.en_embedding and not args.share_embedding:
        print("\t{}: {}".format(
            "Encoder embedding dimension", args.en_embedding_dim))
        print("\t{}: {}".format(
            "Decoder embedding dimension", args.de_embedding_dim))
    elif not args.en_embedding:
        print("\t{}: {}".format(
            "Decoder embedding dimension", args.de_embedding_dim))
    else:
        print("\t{}: {}".format(
            "Shared embedding dimension", args.embedding_dim))
    print("\t{}: {}".format("Attention method", args.attn_method))
    print("\t{}: {}".format("Bidrectional RNN", bool(args.bidirectional)))
    print("\t{}: {}".format("Batch normalization", bool(args.batch_norm)))
    print("{}:".format("TRAINING"))
    print("\t{}: {}".format("Training epochs", args.epochs))
    print("\t{}: {}".format("Batch size", args.batch_size))
    print("\t{}: {}".format("Encoder optimizer", args.en_optimizer))
    print("\t{}: {}".format("Encoder learning_rate", args.en_learning_rate))
    print("\t{}: {}".format("Decoder optimizer", args.de_optimizer))
    print("\t{}: {}".format("Decoder learning rate", args.de_learning_rate))
    if args.split_teacher_forcing:
        print("\t{}: {}".format(
            "Inner teacher forcing ratio", args.inner_teacher_forcing_ratio))
        print("\t{}: {}".format(
            "Inter teacher forcing ratio", args.inter_teacher_forcing_ratio))
        print("\t{}: {}".format(
            "Inner teacher forcing decay rate", args.inner_tf_decay_rate))
        print("\t{}: {}".format(
            "Inter teacher forcing decay rate", args.inter_tf_decay_rate))
        print("\t{}: {}".format(
            "Inner schedule sampling", bool(args.inner_schedule_sampling)))
        print("\t{}: {}".format(
            "Inter schedule sampling", bool(args.inter_schedule_sampling)))
    else:
        print("\t{}: {}".format(
            "Teacher forcing ratio", args.teacher_forcing_ratio))
        print("\t{}: {}".format(
            "Teacher forcing decay rate", args.tf_decay_rate))
        print("\t{}: {}".format(
            "Schedule sampling", bool(args.schedule_sampling)))
    print("\t{}: {}".format("Curriculum learning", bool(args.is_curriculum)))
    if args.verbose_level > 1:
        print("\t{}: {}".format("Padding loss", args.padding_loss))
        print("\t{}: {}".format("EOS loss", args.eos_loss))
    print("\t{}: {}".format("Max gradient norm", args.max_norm))
    print("{}:".format("VERBOSE, VALIDATION AND SAVE"))
    print("\t{}: {}".format("Verbose epochs", args.verbose_epochs))
    print("\t{}: {}".format("Verbose batches", args.verbose_batches))
    print("\t{}: {}".format("Validation epochs", args.valid_epochs))
    print("\t{}: {}".format("Validation batches", args.valid_batches))
    print("\t{}: {}".format("Save epochs", args.save_epochs))
    if args.verbose_level > 1:
        print("\t{}: {}".format(
            "Check memory usage batches", args.check_mem_usage_batches))
    print("\t{}: {}".format("Load old model", bool(args.is_load)))
