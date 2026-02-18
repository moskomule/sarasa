from sarasa.config import AdamW, Config, Data, Evaluate, LRScheduler, Model, Train

config = Config.create(
    model=Model(
        name="llama3",
        hidden_dim=2048,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        rms_eps=1e-5,
        rms_learnable=True,
    ),
    data=Data(tokenizer_path="./tokenizer"),
    lr_scheduler=LRScheduler(
        decay_type="linear",
        warmup_steps=0,
    ),
    optim=AdamW(lr=3e-4),
    train=Train(
        local_batch_size=32,
        global_batch_size=1024,
        use_sac=True,
    ),
    evaluate=Evaluate(
        freq=1000,
        val_size=8192,
    ),
    seed=0,
)
