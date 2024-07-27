# 下面所有命令都是加入reflexion这一observation之后的

# 1.baseline
# baseline 结果放到 results/baseline/AGENT_MODEL_NAME 文件夹下

python run.py  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s_reflexion.json --observation_type grounding  --test_start_idx 0  --test_end_idx 1  --model llama2:7b  --result_dir results/baseline/llama2_7b --save_format_trace_enabled

# 2.retrieval
# 有两种模式：prompt_based 和 embedding_based, retrieval 结果放到 results/retrieval/prompt_based（或embedding_based）/retriever_modeL_NAME@AGENT_MODEL_NAME 文件夹下

# 2.1 prompt_based
python run.py  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s_reflexion.json --observation_type grounding  --test_start_idx 0  --test_end_idx 1  --model llama2:7b  --result_dir results/retrieval/prompt_based/qwen2_7b@llama2_7b --save_format_trace_enabled --retriever_mode 1 --r_instruction_path agent/prompts/jsons/p_cot_retrieval_2s_reflexion.json --r_model qwen2:7b

## 2.2 embedding_based
python run.py  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s_reflexion.json --observation_type grounding  --test_start_idx 0  --test_end_idx 1  --model llama2:7b  --result_dir results/retrieval/embedding_based/gte_qwen2_7b@llama2_7b --save_format_trace_enabled --retriever_mode 2 --r_model_name_or_path /data3/funing/gte-qwen2-7b-instruct --r_tokenizer_name_or_path  /data3/funing/gte-qwen2-7b-instruct