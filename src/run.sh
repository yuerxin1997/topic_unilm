DATA_DIR=/home/hejingbo/topic_unilm/src/data/qg/train
OUTPUT_DIR=/home/hejingbo/topic_unilm/src/fine_tuned_model/
MODEL_RECOVER_PATH=/home/hejingbo/topic_unilm/src/pretrain_model/unilm1-base-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/home/hejingbo/topic_unilm/src/pretrain_model/bert-base-cased-cache
export CUDA_VISIBLE_DEVICES=0,1
python biunilm/run_seq2seq.py --do_train --topic_mode 1 --num_workers 0 \
  --bert_model bert-base-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file train.pa.tok.txt --tgt_file train.q.tok.txt \
  --output_dir ${OUTPUT_DIR}/bert_save/qg_base_idea1 \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_position_embeddings 512 \
  --mask_prob 0.7 --max_pred 48 \
  --train_batch_size 32 --gradient_accumulation_steps 2 \
  --learning_rate 0.00002 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 10

for epoch in {1..10}
do
  DATA_DIR=/home/hejingbo/topic_unilm/src/data/qg/test
  UniLM_MODEL_RECOVER_PATH=/home/hejingbo/topic_unilm/src/fine_tuned_model/bert_save/qg_base_idea1/unilm.${epoch}.bin
  Topic_MODEL_RECOVER_PATH=/home/hejingbo/topic_unilm/src/fine_tuned_model/bert_save/qg_base_idea1/topic.${epoch}.ckpt 
  EVAL_SPLIT=test
  export PYTORCH_PRETRAINED_BERT_CACHE=/home/hejingbo/topic_unilm/src/pretrain_model/bert-base-cased-cache
  export CUDA_VISIBLE_DEVICES=0,1
  # run decoding
  python biunilm/decode_seq2seq.py --bert_model bert-base-cased --new_segment_ids --topic_mode 1 --mode s2s \
    --input_file ${DATA_DIR}/$test.pa.tok.txt --split ${EVAL_SPLIT} --tokenized_input \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --max_seq_length 512 --max_tgt_length 48 \
     --batch_size 16 --beam_size 1 --length_penalty 0
done
