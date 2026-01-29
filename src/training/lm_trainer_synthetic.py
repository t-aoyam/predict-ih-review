import os, pathlib, json, pickle, logging, torch, re

import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (GPT2TokenizerFast, AutoModelForCausalLM, GPT2Config, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling, set_seed, TrainerCallback)
# from trainer_with_syntactic_regularizer import TrainerWithSyntacticRegularizer
# from trainer_with_copying_regularizer import TrainerWithCopyingRegularizer
# from trainer_with_smooth_copying_regularizer import TrainerWithSmoothCopyingRegularizer
# from tokenizers import SentencePieceBPETokenizer
from torch import cuda
logging.basicConfig(level=logging.ERROR)

print("In LMTrainer:")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(cuda.is_available())
print(cuda.device_count())

class LMTrainerSynthetic:
    def __init__(self,
                 output_dir,
                 model_name,
                 data_fp,
                 val_fp,
                 reg_fp,
                 from_hub,
                 config_dict,
                 use_pretrained_tokenizer=True,
                 reg_lambda=None,
                 device=None,
                 report_to=None,
                 smooth=False,
                 # args below will be passed as kwargs (**config['lmtrainer'])
                 seed=None,#42,
                 vocab_size=None,#=20_000,
                 data_size_for_lm=None,#=100_000_000,
                 data_size_for_tokenizer=None,#=5_000_000,
                 target_num_toks_for_lm=None,#=200_000_000,
                 context_length=None,#=128,
                 save_every_n_words=None,#=20_000_000,
                 save_at_n_words=None,
                 layers_to_unfreeze=None,#=["transformer.wte.weight"],
                 ):
        self.reg_data = None
        self.tokenized_val_data = None
        self.chunked_val_data = None
        self.seed = int(seed)
        self.output_dir = output_dir
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.data_fp = data_fp
        self.val_fp = val_fp
        self.reg_fp = reg_fp
        self.corpus = self.corpus_type = data_fp.split(os.path.sep)[-1].split('.')[0].split('-')[0]
        self.use_pretrained_tokenizer = use_pretrained_tokenizer
        self.reg_lambda = reg_lambda
        # self.lang, self.corpus_type = self.corpus.split('-')
        self.data_size_for_lm = data_size_for_lm
        self.data_size_for_tokenizer = data_size_for_tokenizer
        self.target_num_toks_for_lm = target_num_toks_for_lm
        self.save_every_n_words = save_every_n_words
        self.save_at_n_words = save_at_n_words
        self.context_length = context_length
        self.layers_to_unfreeze = layers_to_unfreeze
        self.lm_config_dict = config_dict['lm']
        self.trainer_config_dict = config_dict['lm_training']
        self.simplewiki_title_fp = os.path.join('data', 'simplewiki_titles.json')
        self.device = device
        self.report_to = report_to
        self.smooth = smooth

    def jsonl2data(self, jsonl_fp, ctx_size, data_size):
        data = []
        num_toks = 0
        pbar = tqdm.tqdm(total=data_size)
        with open(jsonl_fp) as f:
            for line in f:
                line = line.strip()
                seq = json.loads(line)
                data.append(seq['input_ids'])
                num_toks += ctx_size
                pbar.update(ctx_size)
                if num_toks >= data_size:
                    break
        return data

    def _load_data_from_hub(self):
        try:
            if self.corpus_type == 'cc100':
                data = load_dataset('cc100', 'en', streaming=True)
                self.data = data['train']
            elif self.corpus_type == 'simplewiki':
                data = load_dataset('rahular/simple-wikipedia', streaming=True)
                self.data = data['train']
            else:
                data = load_dataset(self.corpus_type, streaming=True)
                self.data = data['train']
        except:
            raise IOError(f"{self.corpus_type} is not yet supported as of now.")
    def _load_data_from_machine(self):
        print(self.data_fp)
        print(self.val_fp)
        if self.data_fp.endswith('json'):
            with open(self.data_fp) as f:
                self.tokenized_data = json.load(f)
        elif self.data_fp.endswith('pkl'):
            with open(self.data_fp, 'rb') as f:
                self.tokenized_data = pickle.load(f)
        else:
            raise IOError(f"data type {self.data_fp.split('.')[-1]} not supported")

    def _batchify_for_tokenizer(self, batch_size=1_000):
        total = 0
        batch = []
        for sample in self.data:
            if total >= self.data_size_for_tokenizer:
                return
            text = sample['text'].strip('\n')
            if not text:
                continue
            batch.append(text)
            if len(batch) == batch_size:
                total += batch_size
                yield batch
                batch = []
            print(f"\rBatchifying... {round(total / self.data_size_for_tokenizer, 3) * 100}%", end="")
        print("\n")

    def _chunk_for_lm(self, vectorize=True):
        if vectorize:
            overflow = len(self.tokenized_data) % self.context_length
            if overflow > 0:
                self.tokenized_data = self.tokenized_data[:-overflow]
            self.tokenized_data = torch.tensor(self.tokenized_data, dtype=torch.int).to(self.device)
            self.chunked_data = self.tokenized_data.reshape(-1, self.context_length)
            self.chunked_data = DatasetDict(
                {
                'train': Dataset.from_dict(
                    {'input_ids': self.chunked_data.tolist()}
                )
                }
            )
        else:
            self.chunked_data = []
            for start_idx in range(0, self.data_size_for_lm, self.context_length):
                chunk = self.tokenized_data[start_idx:start_idx + self.context_length]
                self.chunked_data.append(chunk)
                progress = round(start_idx / self.data_size_for_lm, 5) * 100
                print(f"\rChunking train into sequences of length {self.context_length}... {progress}%", end="")
            self.chunked_data = DatasetDict(
                {
                'train': Dataset.from_dict(
                    {'input_ids': self.chunked_data[:-1]}  # dropping the last sequence
                )
                }
            )
        del self.tokenized_data

        if vectorize:
            overflow = len(self.tokenized_val_data) % self.context_length
            if overflow > 0:
                self.tokenized_val_data = self.tokenized_val_data[:-overflow]
            self.tokenized_val_data = torch.tensor(self.tokenized_val_data, dtype=torch.int).to(self.device)
            self.chunked_val_data = self.tokenized_val_data.reshape(-1, self.context_length)
            self.chunked_val_data = Dataset.from_dict(
                    {'input_ids': self.chunked_val_data.tolist()}
                )
        else:
            self.chunked_val_data = []
            for start_idx in range(0, 100000, self.context_length):
                chunk = self.tokenized_val_data[start_idx:start_idx + self.context_length]
                self.chunked_val_data.append(chunk)
                progress = round(start_idx / self.data_size_for_lm, 5) * 100
                print(f"\rChunking val into sequences of length {self.context_length}... {progress}%", end="")
            self.chunked_val_data = Dataset.from_dict(
                {'input_ids': self.chunked_val_data[:-1]}  # dropping the last sequence
            )
        del self.tokenized_val_data
        torch.cuda.empty_cache()

    def process_data(self):
        # Get tokenizer
        if self.use_pretrained_tokenizer:  # Load pretrained tokenizer
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        else:  # Train tokenizer
            print("Training tokenizer")
            self.tokenizer = self._train_tokenizer()
            tokenizer_model_path = os.path.join(self.output_dir, f"{self.model_name}-GPT2TokenizerFast")
            if not os.path.exists(tokenizer_model_path):
                os.mkdir(tokenizer_model_path)
            self.tokenizer.save_pretrained(tokenizer_model_path)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        # Load data (and tokenize if needed)
        # reg
        if self.reg_fp:
            self.reg_data = load_dataset(
                "json", data_files=self.reg_fp, split='train', streaming=True)
            self.reg_data = self.reg_data.with_format('torch')

        # val
        # with open(self.val_fp, 'rb') as f:
        #     self.tokenized_val_data = pickle.load(f)
        self.chunked_val_data = self.jsonl2data(
            self.val_fp,
            data_size=100_000,
            ctx_size=int(re.search(r'c(\d+)', self.model_name)[1])
        )
        self.chunked_val_data = Dataset.from_dict(
                {'input_ids': self.chunked_val_data[:-1]}  # dropping the last sequence
            )

        # train
        if '.' not in self.data_fp:  # HF
            self._load_data_from_hub
            self._tokenize_data()
        elif self.data_fp.endswith('pkl') or self.data_fp.endswith('json'):  # local tokenized data
            self._load_data_from_machine()
        elif self.data_fp.endswith('jsonl'):
            self.chunked_data = load_dataset(
                "json", data_files=self.data_fp, split="train", streaming=True)
            self.chunked_data = self.chunked_data.with_format('torch')
            # self.chunked_data = jsonl2data(self.data_fp)
            # self.chunked_data = DatasetDict(
            #     {
            #         'train': Dataset.from_dict(
            #             {'input_ids': self.chunked_data}  # dropping the last sequence
            #         )
            #     }
            # )
            return
        # Chunk into model's context size if not jsonl
        self._chunk_for_lm()

        print(f'Length of train data={len(self.chunked_data["train"])}')
        print(f'Length of val data={len(self.chunked_val_data)}')

    def train_lm(self):
        # Initialize logger
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        logger.setLevel(logging.INFO)
        set_seed(self.seed)
        self.process_data()

        # Initialize trainer
        logger.info("Initialising GPT-2 from scratch")

        train_vocab_size = re.search(r'v(\d+)_', self.data_fp)[1]
        # val_vocab_size = re.search(r'v(\d+)_', self.val_fp)[1]
        # if 'natural' in self.data_fp:
        #     config = GPT2Config(
        #         # vocab_size=int(train_vocab_size)+int(val_vocab_size)+1,
        #         vocab_size=50257,
        #         n_ctx=self.context_length,
        #         n_positions=self.context_length,
        #         # eos_token_id=self.tokenizer.eos_token_id,
        #         eos_token_id=50256,
        #         **self.lm_config_dict,
        #     )
        # else:  # synthetic data with 10:1 train:val vocab size ratio
        #     config = GPT2Config(
        #         # vocab_size=int(train_vocab_size)+int(val_vocab_size)+1,
        #         vocab_size=int(int(train_vocab_size)*1.1)+1,  # train + val + EOS
        #         n_ctx=self.context_length,
        #         n_positions=self.context_length,
        #         # eos_token_id=self.tokenizer.eos_token_id,
        #         eos_token_id=int(int(train_vocab_size)*1.1),  # if 100+10, it's 0-109. So EOS should be 110
        #         **self.lm_config_dict,
        #     )
        # just follow the data vocab size!
        config = GPT2Config(
            vocab_size=int(train_vocab_size),
            n_ctx=self.context_length,
            n_positions=self.context_length,
            eos_token_id=int(train_vocab_size)-1,
            **self.lm_config_dict,
        )

        print(GPT2Config)
        model = AutoModelForCausalLM.from_config(config)
        model_size = sum(t.numel() for t in model.parameters())

        print(f"Model parameter size: {model_size / 1000 ** 2:.1f}M parameters")
        trainable_params = sum(t.numel() for t in model.parameters() if t.requires_grad==True)
        print(f"Trainable parameter size: {trainable_params / 1000 ** 2:.1f}M parameters")
        toks_per_step = (
                self.trainer_config_dict['per_device_train_batch_size']*\
                cuda.device_count()*\
                self.trainer_config_dict['gradient_accumulation_steps']*\
                self.context_length
        )
        max_steps = int(self.target_num_toks_for_lm/toks_per_step)
        warmup_steps = int(max_steps*0.01)

        # construct base training argument object
        training_args = TrainingArguments(
            report_to=self.report_to,
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            do_train=True,
            do_eval=False,
            do_predict=False,
            max_steps=max_steps,
            save_steps=1_000_000_000_000,
            logging_steps=1_000_000_000_000,
            warmup_steps=warmup_steps,
            **self.trainer_config_dict
        )

        if self.save_at_n_words:
            class CustomSaveCallback(TrainerCallback):
                def __init__(self, steps_to_save):
                    self.steps_to_save = set(steps_to_save)

                def on_step_end(self, args, state, control, **kwargs):
                    # control.should_log = True
                    if state.global_step in self.steps_to_save:
                        control.should_log = True
                        control.should_evaluate = True
                        # Save the model
                        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                        os.makedirs(output_dir, exist_ok=True)
                        kwargs['model'].save_pretrained(output_dir)
                        kwargs['tokenizer'].save_pretrained(output_dir)
                        print(f"Model saved at step {state.global_step}")

            training_args.save_steps = 1_000_000_000_000  # avoid saving besides the custom saving
            training_args.logging_steps = 1_000_000_000_000  # avoid logging besides the custom saving
            save_at_n_steps = [int(int(n_word)/toks_per_step) for n_word in self.save_at_n_words]

        elif self.save_every_n_words:
            training_args.save_steps = int(self.save_every_n_words/toks_per_step)
            training_args.logging_steps = int(self.save_every_n_words/toks_per_step)

        else:
            raise IOError("No saving method specified.")

        if self.reg_lambda:
            if self.reg_fp:
                if self.smooth:
                    training_args.dataloader_drop_last = True
                    trainer = TrainerWithSmoothCopyingRegularizer(
                        model=model,
                        tokenizer=self.tokenizer,
                        args=training_args,
                        data_collator=self.data_collator,
                        train_dataset=self.chunked_data,
                        eval_dataset=self.chunked_val_data,
                        reg_dataset=self.reg_data,
                        reg_lambda=self.reg_lambda,
                        device=self.device
                    )
                else:
                    trainer = TrainerWithCopyingRegularizer(
                        model=model,
                        tokenizer=self.tokenizer,
                        args=training_args,
                        data_collator=self.data_collator,
                        train_dataset=self.chunked_data,
                        eval_dataset=self.chunked_val_data,
                        reg_dataset=self.reg_data,
                        reg_lambda=self.reg_lambda,
                        device=self.device
                    )
            else:
                trainer = TrainerWithSyntacticRegularizer(
                    model=model,
                    tokenizer=self.tokenizer,
                    args=training_args,
                    data_collator=self.data_collator,
                    train_dataset=self.chunked_data,
                    eval_dataset=self.chunked_val_data,
                    reg_lambda=self.reg_lambda,
                    device=self.device
                )
        else:
            if self.data_fp.endswith('.pkl'):
                trainer = Trainer(
                    model=model,
                    tokenizer=self.tokenizer,
                    args=training_args,
                    data_collator=self.data_collator,
                    train_dataset=self.chunked_data['train'],
                    eval_dataset=self.chunked_val_data,
                )
            elif self.data_fp.endswith('.jsonl'):
                trainer = Trainer(
                    model=model,
                    tokenizer=self.tokenizer,
                    args=training_args,
                    data_collator=self.data_collator,
                    train_dataset=self.chunked_data,
                    eval_dataset=self.chunked_val_data,
                )
        if self.save_at_n_words:
            # print(save_at_n_steps)
            # modify the callback_handler attribute instead of directly modifying the trainer attribute
            trainer.callback_handler.callbacks.append(CustomSaveCallback(save_at_n_steps))

        print(f"using: {next(trainer.model.parameters()).device}")
        print(cuda.is_available())  # Should print True if CUDA is available
        print(cuda.device_count())  # Should print the number of GPUs
        print('------------------------------CONFIG------------------------------')
        print("| {:<25}: {:>36}|".format('ctx_size', self.context_length))
        print("| {:<25}: {:>36}|".format('# layers', self.lm_config_dict["n_layer"]))
        print("| {:<25}: {:>36}|".format('# heads', self.lm_config_dict["n_head"]))
        print("| {:<25}: {:>36}|".format('batch size', self.trainer_config_dict["per_device_train_batch_size"]))
        print("| {:<25}: {:>36}|".format('grad acc', self.trainer_config_dict["gradient_accumulation_steps"]))
        print("| {:<25}: {:>36}|".format('total batch', self.trainer_config_dict["per_device_train_batch_size"] *
                                        self.trainer_config_dict["gradient_accumulation_steps"]))
        print("| {:<25}: {:>36}|".format('# max steps', max_steps))
        print("| {:<25}: {:>36}|".format('# warmup steps', warmup_steps))
        if self.data_fp.endswith('.pkl'):
            print("| {:<25}: {:>41}|".format('total ', len(self.chunked_data['train'])*self.context_length))
        print('------------------------------------------------------------------')
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too

    def freeze(self, model):
        if not self.layers_to_unfreeze:
            print("No layers are selected - all layers will remain trainable")
            return model
        for name, param in model.named_parameters():
            if name not in self.layers_to_unfreeze:
                param.requires_grad = False
            else:
                print(f"- {name}")
        print('above layers are set to be trainable (and all other layers are frozen!)')
        return model
