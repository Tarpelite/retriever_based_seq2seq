# path of training data
TRAIN_FILE=~/workspace/data/newsqa/train.json
TRAIN_DOC_FILE=~/workspace/data/newsqa/docs.txt
MODEL_PATH=~/workspace/models/unilm_base_uncased_v1.2/pytorch_model.bin
# folder used to save fine-tuned checkpoints
OUTPUT_DIR=~/workspace/models/newsqa_ft_R
# folder used to cache package dependencies
CACHE_DIR=~/workspace/cache_newsqa_ft_R

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_retrieval_based_seq2seq.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} --train_doc_file ${TRAIN_DOC_FILE} \
  --model_type bert --model_name_or_path ${MODEL_PATH} --config_name bert-base-uncased --tokenizer_name bert-base-uncased\
  --do_lower_case  --max_source_seq_length 800 --max_target_seq_length 16 \
  --per_gpu_train_batch_size 2 --gradient_accumulation_steps 1 \
  --learning_rate 7e-5 --num_warmup_steps 50000 --num_training_steps 300000 --cache_dir ${CACHE_DIR}
