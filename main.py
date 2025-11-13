import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments

import wandb
from my_dataset import DATASETMAPPING
from my_model import get_text_to_text_model
from my_util.llm import compute_metrics, get_data_collator, preprocess_logits_for_metrics
from trainer_callbacks import ShowInfoCallback


@hydra.main(version_base="1.3", config_path="conf/hydra", config_name="config")
def main(cfg: DictConfig):
    run_name = f"{cfg.dataset.name}-{cfg.model.name.replace('/','_').replace('-','_')}-n_epochs:{cfg.train.num_train_epochs}-lr:{cfg.train.learning_rate}-batch_size:{cfg.train.train_batch_size}"
    print(OmegaConf.to_yaml(cfg))
    wandb.init(
        mode=cfg.wandb.mode,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=run_name,
        config=dict(cfg),
    )
    # data
    dataset_name = cfg.dataset.name
    data_load_fn = DATASETMAPPING[dataset_name]
    train_set, validation_set, test_set = data_load_fn()

    # model
    model_name = cfg.model.name
    model, tokenizer = get_text_to_text_model(model_name)
    # in Trainer, algorithm(optimizer, scheduler) will use default AdamW and liner_warmup_scheduler
    # Prepare the training arguments
    training_args = TrainingArguments(
        report_to=["wandb"],
        output_dir=f"./results/{run_name}/{cfg.train.seed}",
        # run_name=f"{cfg.wandb.project}/{run_name}",
        run_name=f"{run_name}",
        per_device_train_batch_size=cfg.train.train_batch_size,
        per_device_eval_batch_size=cfg.train.train_batch_size,
        num_train_epochs=cfg.train.num_train_epochs,
        save_strategy=cfg.train.save_strategy,
        save_total_limit=10,  # limit the number of saved checkpoints
        save_steps=1000,
        eval_strategy=cfg.train.eval_strategy,
        eval_steps=cfg.train.eval_steps,
        eval_accumulation_steps=128,
        logging_steps=10,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=0.1,
        weight_decay=cfg.train.weight_decay,
        remove_unused_columns=False,  # tokenize the dataset on the fly
        label_names=["labels"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg.train.seed,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=validation_set,
        data_collator=get_data_collator(tokenizer),
        callbacks=[ShowInfoCallback()],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Start fine-tuning
    trainer.train()

    eval_result = trainer.evaluate()
    print(f"Evaluation result: {eval_result}")
    wandb.log(eval_result)

    save_dir = "./checkpoint/finetuned-" + run_name.replace("/", "-")
    # Save the fine-tuned model
    trainer.save_model(save_dir)


if __name__ == "__main__":
    main()
