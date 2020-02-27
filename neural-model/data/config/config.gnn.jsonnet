{
  "data": {
    "train_file": "data/preprocessed_data/train-shard-*.tar",
    "dev_file": "data/preprocessed_data/dev.tar",
    "vocab_file": "data/vocab.bpe10000/vocab",
  },
  "encoder": {
    "type": 'GraphASTEncoder',
    "connections": ['top_down', 'bottom_up', 'terminals', 'variable_master_nodes', 'func_root_to_arg'],
    "vocab_file": $['data'].vocab_file,
    "decoder_hidden_size": $['decoder'].hidden_size,
    "gnn": {
        "hidden_size": 128,
        "layer_timesteps": [8],
        "residual_connections": {"0": [0]}
    },
    "node_content_embedding_size": 128,
    "init_with_seq_encoding": false,
    "seq_encoder": {
        "vocab_file": $['data'].vocab_file,
        "source_embedding_size": 256,
        "source_encoding_size": 256,
        "dropout": 0.2,
        "decoder_hidden_size": $['decoder'].hidden_size,
    }
  },
  "decoder": {
    "type": 'AttentionalRecurrentSubtokenDecoder',
    "attention_target": "terminal_nodes",
    "vocab_file": $['data'].vocab_file,
    "hidden_size": 256,
    "ast_node_encoding_size": $['encoder'].gnn.hidden_size
  },
  "train": {
    "batch_size": 2000,
    "eval_batch_size": 500,
    "buffer_size": 5000,
    "log_every": 50,
    "unchanged_variable_weight": 0.1,
    "evaluate_every_nepoch": 5,
    "num_readers": 5,
    "num_batchers": 5
  }
}
