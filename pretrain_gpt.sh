DISTRIBUTED_ARGS=(
    --nproc_per_node $SLURM_GPUS_PER_NODE 
    --nnodes $SLURM_JOB_NUM_NODES
    --node_rank $SLURM_NODEID
    --rdzv_id ${SLURM_JOB_ID:-12345}
    --rdzv_backend c10d
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
)
GPT_MODEL_ARGS+=(
    --num-layers 24 
    --hidden-size 2304 
    --num-attention-heads 24 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --attention-backend unfused
    --make-vocab-size-divisible-by 1024
)
TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 64
    --train-iters 100
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16 
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6 
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 6
    --data-parallel-sharding-strategy optim
    --num-layers-per-virtual-pipeline-stage 2
    --microbatch-group-size-per-virtual-pipeline-stage 2
    --disable-tp-comm-overlap-ag
    --disable-tp-comm-overlap-rs
    --disable-tp-comm-bulk-dgrad
    --disable-tp-comm-bulk-wgrad
    --disable-tp-comm-split-ag
    --disable-tp-comm-split-rs
)
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
)
# DATA_ARGS=(
#     --data-path $DATA_PATH 
#     --vocab-file $VOCAB_FILE 
#     --merge-file $MERGE_FILE 
#     --split 949,50,1
# )
EVAL_AND_LOGGING_ARGS=(
    --log-memory-to-tensorboard
    --log-params-norm
    --log-throughput
    --timing-log-level 2
    --log-energy
    --timing-log-option minmax
    --log-interval 1
    --save-interval 10000 
    --eval-interval 100 
    --save $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)
EVAL_AND_LOGGING_ARGS+=(
    --wandb-project $WANDB_PROJECT
    --wandb-exp-name $WANDB_NAME 
)

$VIRTUAL_ENV/Scripts/torchrun.exe ${DISTRIBUTED_ARGS[@]} $MEGATRON_ROOT_PATH/megatron/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    # ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
