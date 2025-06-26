BASEDIR=/app/output/RESULTS_BATCHSIZE
mkdir -p ${BASEDIR}

#for batch_size in 3072 16384 8192 4096 2048 1024 
for batch_size in 24576
do

for gpu_mem_util in 0.90 0.92 0.95 
do

  for max_num_seqs in 16384 8192 4096 2048 1024 512
  do

  	for max_model_len in 4096 2048 
  	do
		OUTPUT_DIR=${BASEDIR}/REP_1_GPUS_2_BS_${batch_size}_MEMUTIL_${gpu_mem_util}_MAXNUMSEQS_${max_num_seqs}_MAXMODELLEN_${max_model_len}
		echo ${OUTPUT_DIR}
		mkdir -p ${OUTPUT_DIR}
		FILENAME="output.log"
		echo "python3 batched_offline_batching.py --num_replicas 1 --num_gpus 2 --model_name /app/model_storage/ --scheduling round_robin --dataset_path /app/data/open_orca_gpt4_tokenized_llama.sampled_24576.pkl --log_level ERROR --user-conf user.conf --num_samples 2000 --output-log-dir ${OUTPUT_DIR} --batch_size ${batch_size} --gpu_mem_util ${gpu_mem_util}  --max_model_len ${max_model_len}  --max_num_seqs ${max_num_seqs} >& ${OUTPUT_DIR}/${FILENAME}"
		python3 batched_offline_batching.py --num_replicas 1 --num_gpus 2 --model_name /app/model_storage/ --scheduling round_robin --dataset_path /app/data/open_orca_gpt4_tokenized_llama.sampled_24576.pkl --log_level ERROR --user-conf user.conf --num_samples 2000 --output-log-dir ${OUTPUT_DIR} --batch_size ${batch_size} --gpu_mem_util ${gpu_mem_util}  --max_model_len ${max_model_len}  --max_num_seqs ${max_num_seqs} >& ${OUTPUT_DIR}/${FILENAME}
		#sleep 2

	done
  done

done

done
