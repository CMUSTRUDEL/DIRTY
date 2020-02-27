{
  "data": {
    "train_file": "data/preprocessed_data/train-shard-*.tar",
    "dev_file": "data/preprocessed_data/dev.tar",
    "vocab_file": "data/vocab.bpe10000/vocab",
  },
  "encoder":{
    "type": "SequentialEncoder",
    "source_embedding_size": 128,
    "source_encoding_size": 256,
    "vocab_file": $['data'].vocab_file,
    "decoder_hidden_size": $['decoder'].hidden_size,
    "dropout": 0.2,
    "num_layers": 2,
  },
  "decoder": {
    "type": 'AttentionalRecurrentSubtokenDecoder',
    "variable_encoding_size": $['encoder'].source_encoding_size,
    "context_encoding_size": $['encoder'].source_encoding_size,
    "vocab_file": $['data'].vocab_file,
    "hidden_size": 256,
    "attention_target": "terminal_nodes"
  },
  "train": {
    "batch_size": 70000,
    "eval_batch_size": 10000,
    "buffer_size": 5000,
    "log_every": 50,
    "unchanged_variable_weight": 0.1,
    "evaluate_every_nepoch": 5,
    "num_readers": 5,
    "num_batchers": 5
  }
}
