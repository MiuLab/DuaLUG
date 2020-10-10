import os
# import tqdm
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import tokenizers
import multiprocessing
cores_num = multiprocessing.cpu_count()
import pdb

# from NLG
def reorder_class_nlg(li):
    ret = []
    for c in li:
        if c[0] == "B":
            ret.append(c[2:])
    ret.append("NULL")
    return ret

# from NLU
def reorder_class_nlu(li):
    ret = []
    if 'O' in li:
        ret.append('O')
    for c in li:
        if c[0] == "B":
            ret.append("B"+c[1:])
            ret.append("I" + c[1:])
    return ret

def reorder_class(li):
    ret = []
    for c in li:
        if c[0] == "B":
            ret.append(c[2:])
    return ret

class DataEngine(Dataset):
    def __init__(self, data_dir, data_split, with_intent=True, add_eos=True):
        # pdb.set_trace()
        self.tokenizer = tokenizers.BertWordPieceTokenizer(os.path.join(data_dir, "tokenizer-vocab.txt"), add_special_tokens=False)
        # self.tokenizer = tokenizers.BertWordPieceTokenizer(os.path.join(data_dir, "tokenizer-vocab.txt"))
        # self.slot_vocab = [x.strip() for x in open(os.path.join(data_dir, "vocab.slot")).readlines()]
        # self.slot_vocab = reorder_class(self.slot_vocab)
        # self.slot2index = {k:i for i,k in enumerate(self.slot_vocab)}
        self.slot_vocab = [x.strip() for x in open(os.path.join(data_dir, "vocab.slot")).readlines()]
        self.nlu_slot_vocab = reorder_class_nlu(self.slot_vocab)
        self.nlg_slot_vocab = reorder_class_nlg(self.slot_vocab)
        # self.slot2index = {k:i for i,k in enumerate(self.slot_vocab)}
        self.nlu_slot2index = {k:i for i,k in enumerate(self.nlu_slot_vocab)}
        self.nlu_index2slot = {i:k for i,k in enumerate(self.nlu_slot_vocab)}
        
        self.nlg_slot2index = {k:i for i,k in enumerate(self.nlg_slot_vocab)}
        self.nlg_index2slot = {i:k for i,k in enumerate(self.nlg_slot_vocab)}
        
        self.with_intent = with_intent
        self.add_eos = add_eos
        # self.intent_vocab = ['O']
        # if self.with_intent:
        #     self.intent_vocab = [x.strip() for x in open(os.path.join(data_dir, "vocab.intent")).readlines()]
        #     self.intent2index = {k:i for i, k in enumerate(self.intent_vocab)}
        #     self.index2intent = {i:k for i, k in enumerate(self.intent_vocab)}
        self.intent_vocab = [x.strip() for x in open(os.path.join(data_dir, "vocab.intent")).readlines()]
        self.intent2index = {k:i for i, k in enumerate(self.intent_vocab)}
        self.index2intent = {i:k for i, k in enumerate(self.intent_vocab)}

        self.load_data(os.path.join(data_dir, data_split))
        # self.find_multi_refs()
        self.find_multi_refs_mt()

        self.training_set_label_samples = self._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def collate_fn_nlg(self, batch):
        # pdb.set_trace()
        
        max_len = max([len(x["nlg_target"]) for x in batch])
        input_term_lens = []
        input_lens = []
        for x in batch:
            input_term_lens.append(len(x['nlg_inputs'].keys()))
            for k in x['nlg_inputs']:
                input_lens.append(len(x['nlg_inputs'][k]))
        max_value_len = max(input_lens)
        key_value = []
        for i in range(len(batch)):
            batch[i]["nlg_target"] = batch[i]["nlg_target"] + [0] * (max_len - len(batch[i]["nlg_target"]))
            for key in batch[i]['nlg_inputs']:
                batch[i]['nlg_inputs'][key] = batch[i]['nlg_inputs'][key] + [0] * (max_value_len - len(batch[i]['nlg_inputs'][key]))
                key_value.append([key, batch[i]['nlg_inputs'][key]])
        target = torch.tensor([x["nlg_target"] for x in batch])
        flat_keys = torch.tensor([x[0] for x in key_value])
        flat_values = torch.tensor([x[1] for x in key_value])
        if batch[0]["nlg_intent"] is not None:
            intent = torch.tensor([x["nlg_intent"] for x in batch])
        else:
            intent = None
        multi_refs = [x["nlg_multi_refs"] for x in batch]

        # temp sol
        max_len = max([len(x["nlu_inputs"]) for x in batch])
        for i in range(len(batch)):
            batch[i]["nlu_inputs"] = batch[i]["nlu_inputs"] + [0] * (max_len - len(batch[i]["nlu_inputs"]))
            batch[i]["nlu_labels"] = batch[i]["nlu_labels"] + [-1] * (max_len - len(batch[i]["nlu_labels"]))
        inputs = torch.tensor([x["nlu_inputs"] for x in batch])
        labels = torch.tensor([x["nlu_labels"] for x in batch])
        if batch[0]["nlu_intent"] is not None:
            intent = torch.tensor([x["nlu_intent"] for x in batch])
        else:
            intent = None
            # intent = []


        return {
            'slot_key':flat_keys,
            'slot_key_lens':input_term_lens,
            'slot_value':flat_values,
            'slot_value_lens':input_lens,
            'intent':intent,
            'target':target,
            'multi_refs': multi_refs,
            'dual_nlu_inputs': inputs,
            'dual_nlu_labels': labels,
            'dual_nlu_intent': intent
            }

    def collate_fn_nlu(self, batch):
        max_len = max([len(x["nlu_inputs"]) for x in batch])
        for i in range(len(batch)):
            batch[i]["nlu_inputs"] = batch[i]["nlu_inputs"] + [0] * (max_len - len(batch[i]["nlu_inputs"]))
            batch[i]["nlu_labels"] = batch[i]["nlu_labels"] + [-1] * (max_len - len(batch[i]["nlu_labels"]))
        inputs = torch.tensor([x["nlu_inputs"] for x in batch])
        labels = torch.tensor([x["nlu_labels"] for x in batch])
        if batch[0]["nlu_intent"] is not None:
            intent = torch.tensor([x["nlu_intent"] for x in batch])
        else:
            intent = None
            # intent = []
        multi_refs = [x["nlg_multi_refs"] for x in batch]
        return {'inputs': inputs, 'labels': labels, 'intent': intent, 'dual_nlg_multi_refs': multi_refs}

    def iob_id_to_slot_name(self, id):
        if id == 0:
            return 'O'
        else:
            return self.nlu_index2slot[id].split('-')[1]

    def fuzzy_mapping_slots_to_semantic_frame(self, slot_id_seq, word_seq):
        # nlu_index2slot: 0: 'O', odd: 'B-SLOT_NAME', even: 'I-SLOT_NAME'
        assert len(slot_id_seq) == len(word_seq)
        semantic_frame = dict()
        for iob_slot_id, word in zip(slot_id_seq, word_seq): 
            # not 0
            # pdb.set_trace()
            iob_slot_id = int(iob_slot_id)
            if iob_slot_id > 0 and word != 0:
                slot_id = self.nlg_slot2index[self.iob_id_to_slot_name(iob_slot_id)]
                if slot_id not in semantic_frame:
                    semantic_frame[slot_id] = list()
                semantic_frame[slot_id].append(word)
        # check empty
        if len(semantic_frame) == 0:
            slot = self.nlg_slot2index["NULL"]
            semantic_frame.setdefault(slot, [])
            semantic_frame[slot].append(self.tokenizer.token_to_id("[UNK]"))
        return semantic_frame

    def batch_semantic_frame_to_nlg_input(self, batch):
        max_len = max([len(x) for x in batch])
        input_term_lens = []
        input_lens = []
        for x in batch:
            input_term_lens.append(len(x.keys()))
            for k in x:
                input_lens.append(len(x[k]))
        max_value_len = max(input_lens)
        key_value = []
        for i in range(len(batch)):
            for key in batch[i]:
                batch[i][key] = batch[i][key] + [0] * (max_value_len - len(batch[i][key]))
                key_value.append([key, batch[i][key]])
        flat_keys = torch.tensor([x[0] for x in key_value])
        flat_values = torch.tensor([x[1] for x in key_value])
        
        return {
            'slot_key':flat_keys,
            'slot_key_lens':input_term_lens,
            'slot_value':flat_values,
            'slot_value_lens':input_lens}


    def untokenize_semantic_frame(self, semantic_frame):
        untok_sf = dict()
        for k, v in semantic_frame.items():
            untok_sf[self.nlg_index2slot[k]] = self.untokenize(v)
        return untok_sf
    
    def decode_semantic_frame(self, semantic_frame):
        untok_sf = dict()
        for k, v in semantic_frame.items():
            untok_sf[self.nlg_index2slot[k]] = self.decode_word_seq(v)
        return untok_sf

    def ids_to_tokens(self, inputs):
        return [self.tokenizer.id_to_token(x) for x in inputs]

    def tokenize(self, inputs):
        return [self.tokenizer.token_to_id(x) for x in inputs]

    def untokenize(self, inputs):
        return [self.tokenizer.id_to_token(x) for x in inputs]

    def decode_word_seq(self, index_seq):
        return self.tokenizer.decode(index_seq)

    def encode_str_to_tokens(self, str):
        return self.tokenizer.encode(str).tokens

    def encode_str_to_ids(self, str):
        return self.tokenizer.encode(str).ids

    # NOTE: handling padding token: -1
    def tokenize_nlu_slot_seq(self, inputs):
        return [self.nlu_slot2index[i] for i in inputs]

    def untokenize_nlu_slot_seq(self, inputs):
        return [self.nlu_index2slot[i] for i in inputs if i != -1]

    def load_data(self, data_path):
        drop_failed = (data_path.split('/')[-1] == 'valid')
        self._data = []
        with open(data_path, "r") as f:
            id=0
            # for line in tqdm(f.readlines()[:200]):
            for line in tqdm(f.readlines()):
                #NLG
                labels = []
                inputs = []
                raw_ws = []
                raw_ls = []
                if self.with_intent:
                    slot_sent, intent = line.strip().split(" <=> ")
                    if not drop_failed:
                        intent = self.intent2index[intent]
                    else:
                        try:
                            intent = self.intent2index[intent]
                        except:
                            print("ERROR Intent label: %s, skip!"%intent)
                            continue
                else:
                    slot_sent = line.strip()
                    intent = None
                tmp_slot = {}
                for term in slot_sent.split(" "):
                    try:
                        term = term.split(":")
                        label = term[-1]
                        word = ":".join(term[:-1])
                        # assert label[2:] in self.slot_vocab or label[0] == 'O'
                        assert label[2:] in self.nlg_slot_vocab or label[0] == 'O'
                    except:
                        pdb.set_trace()
                    tokenized = self.tokenizer.encode(word)
                    subwords_raw = tokenized.tokens
                    subwords = tokenized.ids
                    for i, subword in enumerate(subwords):
                        # labels.append(self.slot2index[label])
                        # raw_ls.append(label)
                        if label[0] != "O":
                            # slot = self.slot2index[label[2:]]
                            slot = self.nlg_slot2index[label[2:]]
                            tmp_slot.setdefault(slot, [])
                            tmp_slot[slot].append(subword)
                        inputs.append(subword)
                        raw_ws.append(subwords_raw[i])
                    if len(tmp_slot.keys()) == 0:
                        slot = self.nlg_slot2index["NULL"]
                        tmp_slot.setdefault(slot, [])
                        tmp_slot[slot].append(self.tokenizer.token_to_id("[UNK]"))
                if self.add_eos:
                    inputs.append(self.tokenizer.token_to_id('[SEP]'))
                    raw_ws.append('[SEP]')
                # labels.append(tmp_slot)
                # self._data.append({'nlg_target': inputs, 'nlg_inputs': tmp_slot, 'nlg_intent': intent, 'nlg_raw_words': raw_ws})
                nlg_data = {'nlg_target': inputs, 'nlg_inputs': tmp_slot, 'nlg_intent': intent, 'nlg_raw_words': raw_ws}
                
                #NLU
                labels = []
                inputs = []
                raw_ws = []
                raw_ls = []
                if self.with_intent:
                    slot_sent, intent = line.strip().split(" <=> ")
                    intent = self.intent2index[intent]
                else:
                    slot_sent = line.strip()
                    intent = None
                for term in slot_sent.split(" "):
                    try:
                        term = term.split(":")
                        label = term[-1]
                        word = ":".join(term[:-1])
                    except:
                        pdb.set_trace()
                    
                    # pdb.set_trace()

                    # assert label in self.slot_vocab
                    assert label in self.nlu_slot_vocab
                    tokenized = self.tokenizer.encode(word)
                    subwords_raw = tokenized.tokens
                    subwords = tokenized.ids
                    for i, subword in enumerate(subwords):
                        # labels.append(self.slot2index[label])
                        labels.append(self.nlu_slot2index[label])
                        raw_ls.append(label)
                        if i == 0 and label[0] == "B":
                            label = "I" + label[1:]
                        inputs.append(subword)
                        raw_ws.append(subwords_raw[i])
                if self.add_eos:
                    inputs.append(self.tokenizer.token_to_id('[SEP]'))
                    raw_ws.append('[SEP]')
                    labels.append(-1)
                    raw_ls.append("None")
                # self._data.append({'nlu_inputs': inputs, 'nlu_labels': labels, 'nlu_intent': intent, 'nlu_raw_words': raw_ws, 'nlu_raw_labels': raw_ls})
                self._data.append({
                    'id':id,
                    'nlg_target': nlg_data['nlg_target'],
                    'nlg_inputs': nlg_data['nlg_inputs'],
                    'nlg_intent': nlg_data['nlg_intent'],
                    'nlg_raw_words': nlg_data['nlg_raw_words'],
                    'nlg_multi_refs': [nlg_data['nlg_target']],
                    'nlu_inputs': inputs,
                    'nlu_labels': labels,
                    'nlu_intent': intent,
                    'nlu_raw_words': raw_ws,
                    'nlu_raw_labels': raw_ls
                    })

                id += 1
    
    def find_multi_refs(self):
        self.record = dict()
        for id1, data1 in enumerate(tqdm(self._data)):
            self.record[id1] = [id1]
            for id2, data2 in enumerate(self._data):
                if id1 == id2:
                    continue
                if data1['nlg_inputs'] == data2['nlg_inputs'] and \
                    data1['nlg_intent'] == data2['nlg_intent']:
                    # if id1 not in record:
                    #     record[id1] = list()
                    self.record[id1].append(id2)
        
        for id, ref_ids in self.record.items():
            for ref_id in ref_ids:
                self._data[id]['nlg_multi_refs'].append(self._data[ref_id]['nlg_target'])

    def subprocess_find_multi_refs(self, idx, data_chunk, record):
        # print(len(self._data))
        # pass
        # """
        # for _, data1 in enumerate(data_chunk):
        for id1, data1 in enumerate(tqdm(data_chunk, position=idx)):
            # self.record[data1['id']] = [data1['id']]
            for id2, data2 in enumerate(self._data):
                if data1['id'] == data2['id']:
                    continue
                if data1['nlg_inputs'] == data2['nlg_inputs'] and \
                    data1['nlg_intent'] == data2['nlg_intent']:
                    # if id1 not in record:
                    #     record[id1] = list()
                    # self.record[data1['id']].append(data2['id'])
                    # data_chunk[id1]['nlg_multi_refs'].append(data2['nlg_target'])
                    data1['nlg_multi_refs'].append(data2['nlg_target'])
            record.append(data1)
            # print(data1)
            # print(record)
        # for id, ref_ids in self.record.items():
        #     for ref_id in ref_ids:
        #         self._data[id]['nlg_multi_refs'].append(self._data[ref_id]['nlg_target'])
        # """


    def divide_data(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n] 

    def find_multi_refs_mt(self):
        print("Finding multi-references...")
        data_chunk_list = list()
        for item in self.divide_data(self._data, int(len(self._data)/cores_num)):
            data_chunk_list.append(item)

        record = multiprocessing.Manager().list()
        jobs = []
        for i in range(0, cores_num):
            process = multiprocessing.Process(target=self.subprocess_find_multi_refs, args=(i, data_chunk_list[i], record))
            jobs.append(process)

        # Start the processes (i.e. calculate the random number lists)
        for j in jobs:
            j.start()

        # Ensure all of the processes have finished
        for j in jobs:
            j.join()

        self._data = [x for x in record]
        record = None

        # for x in self._data:
        #     if len(x['nlg_multi_refs']) > 1:
        #         print("!")

        # pdb.set_trace()
        # for id, ref_ids in self.record.items():
        #     for ref_id in ref_ids:
        #         self._data[id]['nlg_multi_refs'].append(self._data[ref_id]['nlg_target'])

class DataEngineSplit(Dataset):
    def __init__(self, input_data, output_labels, refs, sf_data, input_attr_seqs):
        super(DataEngineSplit, self).__init__()
        self.input_data = input_data
        self.output_labels = output_labels
        self.refs = refs
        self.sf_data = sf_data
        self.input_attr_seqs = input_attr_seqs

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return (
            self.input_data[idx],
            self.output_labels[idx],
            self.refs[idx],
            self.sf_data[idx],
            self.input_attr_seqs[idx]
        )



if __name__ == "__main__":
    dataset = DataEngine("../data/snips", "test")
    """
                self.train_nlg_data_loader = DataLoader(
                    train_data_engine,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    drop_last=True,
                    collate_fn=train_data_engine.collate_fn_nlg,
                    pin_memory=True)
    """
    # d = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=DataEngine.collate_fn_nlg)
    # for b in d:
    #     break
    pdb.set_trace()
