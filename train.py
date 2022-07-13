import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DefaultDataCollator,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, DatasetDict
from models import T5VAEForConditionalGeneration
from trainers import T5VAETrainer
from utils import random_masking

import os
os.environ["WANDB_DISABLED"] = "true"

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='t5-small')
    model_type: Optional[str] = field(default='t5-small')
    config_name: Optional[str] = field(default='t5-small')
    tokenizer_name: Optional[str] = field(default='t5-small')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Customized arguments
    vae_latent_size: int = field(default=128)
    vae_k: float = field(default=0.0025)
    vae_x0: int = field(default=2500)
    vae_annealing_fn: str = field(default='logistic')

@dataclass
class OurDataArguments:
    # Huggingface's original arguments. 
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_length: int = field(default=258)

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Huggingface's original arguments. 
    output_dir: str = field(default='./models')
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=10000)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)

def main():

    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_datalcasses()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # additional config and tokenizers
    config_kwargs = {
            "latent_size": model_args.vae_latent_size, 
            "k": model_args.vae_k,
            "x0": model_args.vae_x0,
            "annealing_fn": model_args.vae_annealing_fn,
            "output_hidden_states": True,
    }
    tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir, 
            "use_fast": model_args.use_fast_tokenizer
    }

    # init
    config = AutoConfig.from_pretrained(model_args.config_name)
    config.update(config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    model = T5VAEForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            tokenizer=tokenizer
    )

    # Dataset 
    def prepare_dataset(examples):

        source = [src.split('\t')[1] for src in examples['source']]
        target = [tgt.split('\t')[1] for tgt in examples['target']]

        # random masking
        # source = random_masking(source, '<unk>')

        # source tokenize
        features = tokenizer(
                source,
                truncation=True,
                max_length=data_args.max_length,
        )

        # target tokenize
        labels = tokenizer(
                target,
                truncation=True,
                max_length=data_args.max_length,
        )

        # merge together
        features['labels'] = labels.input_ids

        return features

    ## Loading form hf dataset
    dataset_source = load_dataset('text', data_files={
        'train': 'data/canard/train.history.ntr.tsv', 
        'dev': 'data/canard/dev.history.ntr.tsv'
    }).rename_column('text', 'source')

    dataset_target = load_dataset('text', data_files={
        'train': 'data/canard/train.rewrite.tsv', 
        'dev': 'data/canard/dev.rewrite.tsv',
    })

    train_dataset = dataset_source['train'].add_column('target', dataset_target['train']['text'])
    dev_dataset = dataset_source['dev'].add_column('target', dataset_target['dev']['text'])

    ## Preprocessing
    train_dataset = train_dataset.map(
            function=prepare_dataset, 
            remove_columns=['source', 'target'],
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=not data_args.overwrite_cache,
            batched=True,
    )
    dev_dataset = dev_dataset.map(
            function=prepare_dataset, 
            remove_columns=['source', 'target'],
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=not data_args.overwrite_cache,
            batched=True,
    )

    ## data collator
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt'
    )

    # Trainer
    trainer = T5VAETrainer(
            model=model, 
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator
    )
    
    # ***** strat training *****
    results = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    return results

if __name__ == '__main__':
    main()
