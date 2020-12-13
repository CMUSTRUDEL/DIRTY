{
  "data": {
    "train_file": "data1/train-shard-*.tar",
    "dev_file": "data1/dev-*.tar",
    "test_file": "data1/test.tar",
    "vocab_file": "data1/vocab.bpe10000name",
    "typelib_file": "data1/typelib.json",
    "max_src_tokens_len": 510,
    "args": false,
    "retype": false,
    "rename": true,
  },
  "encoder":{
    "type": "XfmrSequentialEncoder",
    "source_embedding_size": 256,
    "hidden_size": 256,
    "vocab_file": $['data'].vocab_file,
    "dropout": 0.1,
    "num_layers": 2,
  },
  "decoder": {
    "type": 'XfmrDecoder',
    "vocab_file": $['data'].vocab_file,
    "typelib_file": "data1/typelib.json",
    "target_embedding_size": $['encoder'].source_embedding_size,
    "hidden_size": $['encoder'].hidden_size,
    "dropout": 0.1,
    "num_layers": 2,
    "mem_mask": "none",
  },
  "mem_encoder":{
    "type": "XfmrMemEncoder",
    "source_embedding_size": 128,
    "hidden_size": 128,
    "vocab_file": $['data'].vocab_file,
    "dropout": 0.1,
    "num_layers": 2,
  },
  "mem_decoder": {
    "type": 'SimpleDecoder',
    "vocab_file": $['data'].vocab_file,
    "hidden_size": $['mem_encoder'].hidden_size,
  },
  "train": {
    "batch_size": 64,
    "eval_batch_size": 64,
    "max_epoch": 100,
    "lr": 1e-3,
    "patience": 10,
  },
  "test": {
    "pred_file": "pred_xfmr.json",
  }
}
