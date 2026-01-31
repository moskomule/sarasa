from sarasa.config import FSDP, AdamW, Config, Data, LRScheduler, Model, Train

config = Config.create(
    model=Model(
        hidden_dim=2048,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        name="llama3",
    ),
    train=Train(
        local_batch_size=32,
        global_batch_size=256,
        dtype="bfloat16",
        use_sac=True,
    ),
    data=Data(tokenizer_path="./tokenizer"),
    lr_scheduler=LRScheduler(
        decay_type="linear",
        warmup_steps=0,
    ),
    optim=AdamW(lr=3e-4),
    distributed=FSDP(),
    seed=12,
)
