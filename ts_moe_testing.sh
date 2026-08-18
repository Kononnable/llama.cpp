#!/bin/bash

rpc=(
#    "172.18.10.2:50052"
#    "192.168.2.2:50052"
    "192.168.4.2:50052"
)
scenarios=({A..C})
split_5_5=(
    "/models/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    "/models/qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
    "/models/gemma-26B-A4B/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
)
split_6_4=(
    "/models/Qwen3.5-122B-A10B/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"
    "/models/GLM4.5-Air/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf"
    "/models/Laguna-S-2.1/Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf"
    "/models/Qwen3-Coder-Next/Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf"
)

mkdir -p ./results

for model in "${split_5_5[@]}"; do
    filename=$(basename "$model" .gguf)
    for i in "${!rpc[@]}"; do

        echo "${filename} ${scenarios[$i]} layer"
        build/bin/llama-completion \
            -lv 3 --fit off -c 7000 --seed 466805893 \
            -m "$model" -no-cnv -f ./prompt_1600.txt -n 1000 \
            -lm dio -sm layer --rpc ${rpc[$i]} -ts 0,1 \
            -ot "blk\\.[0-9]?[02468]\\.ffn_(up|down|gate|gate_up)_(ch|)exps=RPC0[${rpc[$i]}],.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU" \
            2>&1 | tee "./results/${filename}_layer_${scenarios[$i]}.txt"

        echo "${filename} ${scenarios[$i]} tensor"
        build/bin/llama-completion \
            -lv 3 --fit off -c 7000 --seed 466805893 \
            -m "$model" -no-cnv -f ./prompt_1600.txt -n 1000 \
            -lm dio -sm tensor --rpc ${rpc[$i]} -ts 1,5e,5e \
            2>&1 | tee "./results/${filename}_tensor_${scenarios[$i]}.txt"
    done
done
for model in "${split_6_4[@]}"; do
    filename=$(basename "$model" .gguf)
    for i in "${!rpc[@]}"; do

        echo "${filename} ${scenarios[$i]} layer"
        build/bin/llama-completion \
            -lv 3 --fit off -c 7000 --seed 466805893 \
            -m "$model" -no-cnv -f ./prompt_1600.txt -n 1000 \
            -lm dio -sm layer --rpc ${rpc[$i]} -ts 0,1 \
            -ot "blk\\.[0-9]?[012468]\\.ffn_(up|down|gate|gate_up)_(ch|)exps=RPC0[${rpc[$i]}],.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU" \
            2>&1 | tee "./results/${filename}_layer_${scenarios[$i]}.txt"

        echo "${filename} ${scenarios[$i]} tensor"
        build/bin/llama-completion \
            -lv 3 --fit off -c 7000 --seed 466805893 \
            -m "$model" -no-cnv -f ./prompt_1600.txt -n 1000 \
            -lm dio -sm tensor --rpc ${rpc[$i]} -ts 1,6e,4e \
            2>&1 | tee "./results/${filename}_tensor_${scenarios[$i]}.txt"
    done
done
