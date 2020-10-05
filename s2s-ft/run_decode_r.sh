# path of the fine-tuned checkpoint
MODEL_PATH=/data/question_generation/models/newsqa_ft_with_retrieval/ckpt-32000
SPLIT=dev
# input file that you would like to decode
INPUT_JSON=/data/question_generation/newsqa/${SPLIT}.json
DOC_FILE=/data/question_generation/newsqa/docs.txt
CACHE_DOC_PATH=${MODEL_PATH}/cache_newsqa_doc

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python decode_retrieval_based_seq2seq.py \
  --model_type bert --tokenizer_name bert-base-uncased --input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 816 --max_tgt_length 16 --batch_size 1 --beam_size 1 --doc_file ${DOC_FILE} --top_k 5 --cache_feature_file ${CACHE_DOC_PATH} \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."
