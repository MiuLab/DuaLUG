from tokenizers import BertWordPieceTokenizer
import tqdm
tokenizer = BertWordPieceTokenizer()
tokenizer.train(["all.raw"], vocab_size=10000)
# with open("all.raw") as f:
#     for line in tqdm.tqdm(f.readlines()):
#         tokenizer.add_tokens(line.strip().split(" "))
tokenizer.save(".", "tokenizer")

t = BertWordPieceTokenizer("tokenizer-vocab.txt", add_special_tokens=False)