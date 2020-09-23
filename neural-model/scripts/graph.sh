python exp.py \
    train \
    --cuda \
    --work-dir=exp_runs/dire.graph \
    --extra-config='{ "data": {"train_file": ["data/preprocessed_data_small/train-shard-0.tar", "data/preprocessed_data_small/train-shard-1.tar", "data/preprocessed_data_small/train-shard-2.tar"], "dev_file": "data/preprocessed_data_small/dev.tar" }, "decoder": { "input_feed": false, "tie_embedding": true }, "train": { "evaluate_every_nepoch": 5, "max_epoch": 200 } }' \
    data/config/config.gnn.jsonnet
