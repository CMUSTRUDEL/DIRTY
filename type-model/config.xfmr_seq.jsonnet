{
  "data": {
    "train_file": "data/train-shard-*.tar",
    "dev_file": "data/dev.tar",
    "vocab_file": "data/vocab.bpe10000",
    "max_src_tokens_len": 510,
  },
  "encoder":{
    "type": "XfmrSequentialEncoder",
    "source_embedding_size": 256,
    "source_encoding_size": 256,
    "vocab_file": $['data'].vocab_file,
    "decoder_hidden_size": $['decoder'].hidden_size,
    "dropout": 0.1,
    "num_layers": 2,
  },
  "decoder": {
    "type": 'SimpleDecoder',
    "vocab_file": $['data'].vocab_file,
    "hidden_size": 256,
  },
  "train": {
    "batch_size": 64,
    "eval_batch_size": 64,
    "patience": 10,
    "max_epoch": 60,
    "lr": 1e-3,
  }
}
