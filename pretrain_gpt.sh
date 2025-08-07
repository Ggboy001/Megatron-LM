ARCH_ARGS="\
    --num-layers 4 \
    
    --hidden-size 128 \
    --ffn-hidden-size 128 \
    --num-attention-heads 4 \
    --seq-length 16 \
    --max-position-embeddings 16 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rotary-percent 1.0 \
    --use-rope-scaling \
    --rope-scaling-factor 32 \
    --apply-query-key-layer-scaling \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --make-vocab-size-divisible-by 128 \
    --disable-bias-linear \
    --no-bias-swiglu-fusion \
    --no-gradient-accumulation-fusion \
"

PRECISION_ARGS="\
    --attention-softmax-in-fp32 \
    --bf16 \
    --fp8-format hybrid \
"

TRAINING_ARGS="\
    --clip-grad 1.0 \
    --init-method-std 0.02 \
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --train-iters 1 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --use-mcore-models \
    --no-gradient-accumulation-fusion \
    --transformer-impl transformer_engine \
"

IO_ARGS="\
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 50257 \
    --split 949,50,1 \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --save /workspace/checkpoints/ \
    --load /workspace/checkpoints/ \
"


DISTRIBUTED_ARGS="\
--nproc_per_node 2 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000 \
"

PARALLEL_ARGS="\
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 2 \
"

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $DISTRIBUTED_ARGS \
    $HOME/sly/code/Megatron-LM/pretrain_gpt.py \
    $PARALLEL_ARGS $ARCH_ARGS $PRECISION_ARGS $TRAINING_ARGS $IO_ARGS
