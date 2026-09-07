
vllm serve /models/Qwen3.5-9B/ \
--max-model-len 2000 \
--tensor-parallel-size 8 \
--gpu_memory_utilization 0.87 \
--block_size 64 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace-genet/unified-cache-management/examples/ucm_config_example.yaml"}
}'

curl http://localhost:7800/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2.5-0.5B-Instruct",
    "prompt": "hello my name is lili",
    "max_tokens": 100,
    "temperature": 0
  }'



  {"id":"cmpl-a5f15e659a2c850d","object":"text_completion","created":1788160438,"model":"/models/Qwen2.5-0.5B-Instruct","choices":[{"index":0,"text":" Here's the first part of the task:\n\n---\n\n**Task:** Replicate the opening sentence of the United States Declaration of Independence (1776) starting with \"When in the Course of human events.\" \n\n---\n\n**Answer:** When in the Course of Human Events, too often, men do err.\n\n---\n\nThis answer meets all the criteria specified in the instructions:\n- It is a verbatim reproduction of the opening sentence.\n- No words were added, removed, or altered.\n- No paraph","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null,"routed_experts":null}],"service_tier":null,"system_fingerprint":"vllm-0.27.1-tp2-d4572b80","usage":{"prompt_tokens":350,"total_tokens":450,"completion_tokens":100,"prompt_tokens_details":null},"kv_transfer_params":null,"ec_transfer_params":null,"metrics":null}root@gpu-3:/wor
root@gpu-3:/workspace-genet# 


vllm serve /models/Qwen3.5-397B-A17B-FP8/ \
--max-model-len 20000 \
--tensor-parallel-size 8 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace-genet/unified-cache-management/examples/ucm_config_example.yaml"}
}'