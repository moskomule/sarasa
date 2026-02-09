from sarasa.config import AdamW, Config, Data, Evaluate, LRScheduler, Model, Train

config = Config.create(
    model=Model(
        name="nanochat_gpt",
        num_layers=12,
        qk_norm=True,
    ),
    data=Data(tokenizer_path="./tokenizer"),
    lr_scheduler=LRScheduler(
        decay_type="linear",
        warmup_steps=0,
    ),
    optim=AdamW(lr=3e-4),
    train=Train(
        local_batch_size=16,
        global_batch_size=256,
    ),
    evaluate=Evaluate(
        freq=1000,
        val_size=8192,
    ),
    seed=0,
)
