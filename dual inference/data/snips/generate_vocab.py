from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer()
tokenizer.train(["all.raw"], vocab_size=10000)
tokenizer.save(".", "tokenizer")

t = BertWordPieceTokenizer("tokenizer-vocab.txt", add_special_tokens=False)
# Initialize
# bpe = models.BPE.from_files("tokenizer-vocab.json", "tokenizer-merges.txt")
# t = ByteLevelBPETokenizer(lowercase=True)
# t._tokenizer.model = bpe