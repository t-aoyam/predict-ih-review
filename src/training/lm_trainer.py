import os, pathlib, json, pickle, logging, torch

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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(cuda.is_available())
print(f"# available GPUs: {str(cuda.device_count())}")

def jsonl2data(jsonl_fp, data_size=10_000_000, ctx_size=1_024):
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


class LMTrainer:
    def __init__(self,
                 output_dir,
                 model_name,
                 data_fp,
                 val_fp,
                 reg_fp,
                 from_hub,
                 config_dict,
                 model_train=None,
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
        self.seed = seed
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
        self.model_train = model_train
        self.device = device
        self.report_to = report_to
        self.smooth = smooth

        print(f"# current device in LMTrainer: {device}")

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

    def _train_tokenizer(self):
        tokenizer = SentencePieceBPETokenizer()
        batches = self._batchify_for_tokenizer()
        tokenizer.train_from_iterator(batches,
                                      vocab_size=self.vocab_size,
                                      min_frequency=2,
                                      special_tokens=['<|endoftext|>'])
        gpt_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer,
                                          model_max_length = self.context_length,
                                          special_tokens = ['<|endoftext|>'])
        return gpt_tokenizer

    def _tokenize_data(self):
        self.tokenized_data = []
        if self.corpus_type == 'simplewiki':
            with open(self.simplewiki_title_fp) as f:
                simplewiki_titles = json.load(f)
                simplewiki_titles = set(simplewiki_titles)
        data = self.data
        num_toks = 0
        for sample in data:
            text = sample['text'].strip('\n').strip()
            if self.corpus_type == 'cc100':  # new document signaled by '\n\n'
                if not text:
                    self.tokenized_data.append(self.tokenizer.eos_token_id)
                    continue
            elif self.corpus_type == 'simplewiki':  # new document signaled by the title of the page
                if text in simplewiki_titles:
                    self.tokenized_data.append(self.tokenizer.eos_token_id)
                    continue

            outputs = self.tokenizer(
                text,
                truncation=False
            )
            self.tokenized_data.extend(outputs['input_ids'])
            num_toks += len(outputs['input_ids'])
            print(f"\rTokenizing... {round(num_toks / self.data_size_for_lm, 5) * 100}%", end="")
            if num_toks >= self.data_size_for_lm:
                break
        if self.corpus_type == 'simplewiki':
            simplewiki_titles.clear()  # clear

    def _chunk_for_lm(self, vectorize=True, val_only=False):
        if not val_only:
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
        with open(self.val_fp, 'rb') as f:
            self.tokenized_val_data = pickle.load(f)

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
            self._chunk_for_lm(val_only=True)
            return
        # Chunk into model's context size
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
        if not self.model_train:  # need to initialize the model
            logger.info("Initialising GPT-2 from scratch")

            config = GPT2Config(n_ctx=self.context_length,
                                n_positions=self.context_length,
                                eos_token_id=self.tokenizer.eos_token_id,
                                **self.lm_config_dict,
                                )
            print(GPT2Config)
            model = AutoModelForCausalLM.from_config(config)
        else:
            logger.info("Loading the pretrained GPT-2")
            model = AutoModelForCausalLM.from_pretrained(self.model_train)
            model.resize_token_embeddings(new_num_tokens=0)  # reset the embeddings
            model.resize_token_embeddings(new_num_tokens=self.vocab_size)  # set the embeddings to the L2 vocab size
            model = self.freeze(model)
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
                        # kwargs['tokenizer'].save_pretrained(output_dir)
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

        total_params = sum(p.numel() for p in model.parameters())
        embed_params = sum(
            p.numel() for n, p in model.named_parameters()
            if "embed" in n or "wte" in n or "word_embeddings" in n
        )

        print(f"using: {next(trainer.model.parameters()).device}")
        print(cuda.is_available())  # Should print True if CUDA is available
        print(cuda.device_count())  # Should print the number of GPUs
        print('------------------------------CONFIG------------------------------')
        print("| {:<25}: {:>36}|".format('ctx_size', self.context_length))
        print("| {:<25}: {:>36}|".format('# layers', self.lm_config_dict["n_layer"]))
        print("| {:<25}: {:>36}|".format('# heads', self.lm_config_dict["n_head"]))
        print("| {:<25}: {:>36}|".format('# hidden dim', self.lm_config_dict["n_embd"]))
        print("| {:<25}: {:>36}|".format('# params (w/o embed)', (total_params - embed_params)))
        print("| {:<25}: {:>36}|".format('# params (total)', total_params))
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
