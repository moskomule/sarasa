from sarasa.config import Config, Data, LRScheduler, Model, Train

config = Config(
    model=Model(num_layers=12),
    train=Train(
        local_batch_size=16,
        global_batch_size=256,
        dtype="bfloat16",
    ),
    data=Data(tokenizer_path="./tokenizer"),
    lr_scheduler=LRScheduler(
        decay_type="linear",
        warmup_steps=0,
    ),
    seed=12,
)
