{
  "data": {
    "train_file": "data/preprocessed_data/train-shard-*.tar",
    "dev_file": "data/preprocessed_data/dev.tar",
    "vocab_file": "data/vocab.bpe10000/vocab",
  },
  "encoder": {
    "type": "HybridEncoder",
    //"type": "SequentialEncoder",
    //"type": "GraphASTEncoder",
    "graph_encoder": {
        "vocab_file": $['data'].vocab_file,
        "node_syntax_type_embedding_size": 64,
        "node_type_embedding_size": 64,
        "node_content_embedding_size": 128,
        "dropout": 0.2,
        "connections": ['top_down', 'bottom_up', 'terminals', 'variable_master_nodes', 'func_root_to_arg'],
        "gnn": {
          "hidden_size": 128,
          "layer_timesteps": [8],
          "residual_connections": {'0': [0]}
        },
        "decoder_hidden_size": $['decoder'].hidden_size,
    },
    "seq_encoder": {
        "vocab_file": $['data'].vocab_file,
        "source_embedding_size": 256,
        "source_encoding_size": 256,
        "dropout": 0.2,
        "decoder_hidden_size": $['decoder'].hidden_size,
        "num_layers": 2,
    },
    "source_encoding_size": 256,
    "hybrid_method": "linear_proj" // "linear_proj"
  },
  "decoder": {
    "type": 'AttentionalRecurrentSubtokenDecoder',
    "attention_target": "terminal_nodes",
    "vocab_file": $['data'].vocab_file,
    "hidden_size": 256,
    "variable_encoding_size": 256,
    "context_encoding_size": 256
  },
  "train": {
    "batch_size": 80000,
    "eval_batch_size": 8000,
    "buffer_size": 5000,
    "log_every": 50,
    "unchanged_variable_weight": 0.1,
    "evaluate_every_nepoch": 5,
    "num_readers": 5,
    "num_batchers": 5
  }
}
