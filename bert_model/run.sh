export BERT_BASE_DIR=./

transformers-cli convert --model_type bert --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt --config $BERT_BASE_DIR/bert_config.json --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin

mv bert_config.json config.json
