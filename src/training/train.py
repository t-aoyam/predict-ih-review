import os, argparse, glob, json, re, math
# os.environ['HF_HOME'] = os.path.join(pathlib.Path(__file__).parent.resolve(), 'models')
# os.environ['CURL_CA_BUNDLE'] = ''  # if SSL Error
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use GPU 0 and GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(
        raw_data, tokenized_data, val_fp, reg_fp, output_dir, config_dict,
        model_name, reg_lambda, device, report_to, smooth):

    for cat in config_dict:
        for key in config_dict[cat]:
            val = config_dict[cat][key]
            if type(val) == str and re.match(r"\d+\.?\d+e-?\d+", val) is not None:
                config_dict[cat][key] = float(val)
            elif type(val) in [list, float] or not val.replace('_', '').isdigit():
                config_dict[cat][key] = val
            else:
                config_dict[cat][key] = int(val)

    if raw_data:
        from_hub = True
        data_fp = raw_data
    elif tokenized_data:
        from_hub = False
        data_fp = tokenized_data
    elif segmented_data:
        from_hub = False
        data_fp = segmented_data

    print('\n' + '=' * 100 + f'Training GPT-2 on {data_fp}) from scratch...')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    trainer = LMTrainer(output_dir=output_dir,
                        model_name=model_name,
                        data_fp=data_fp,
                        val_fp=val_fp,
                        reg_fp=reg_fp,
                        config_dict=config_dict,
                        from_hub=from_hub,
                        reg_lambda=reg_lambda,
                        device=device,
                        report_to=report_to,
                        smooth=smooth,
                        **config_dict['lm_trainer']
                        )
    trainer.train_lm()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to train mini gpt2 model")
    parser.add_argument("--raw_data", default=None, help="name of training corpus")
    parser.add_argument("--tokenized_data", default=None, help="pretokenized .pkl data for training")
    parser.add_argument("--segmented_data", default=None, help="pretokenized and segmented .jsonl data for training")
    parser.add_argument("--validation_data", default=os.path.join('data', 'pile_100k_tokens_gpt.pkl'),
                        help="pretokenized .pkl data for validation")
    parser.add_argument("--regularization_data", default=None,
                        help="pretokenized .jsonl data for regularization")
    parser.add_argument("--use_wandb", action="store_true", help="should I use wandb for logging?")
    parser.add_argument("--gpu", type=str, default='0', help="which GPU to use; 0 or 1; choose wisely based on GPU usage!")
    parser.add_argument("--config_fp", required=True, help="fp to the .json config file for tokenizer and lm training")
    parser.add_argument("--n_layer", required=True, help="number of layers")
    parser.add_argument("--n_head", required=True, help="number of heads")
    parser.add_argument("--n_embd", default=768, help="hidden dimension, GPT2 default=768")
    parser.add_argument("--t_block", default='mlp', help="type of transformer block")
    parser.add_argument("--ctx_size", required=True, help="context size")
    parser.add_argument("--batch_size", required=True, help="batch size")
    parser.add_argument("--gacc", required=True, help="gradient accumulation, default=1")
    parser.add_argument("--reg_lambda", type=float, default=0, help="lambda for {syntactic|copying} regularizer")
    parser.add_argument("--smooth", action="store_true", help="regularize every step, default=False")

    args = parser.parse_args()
    raw_data, tokenized_data, segmented_data, validation_data, regularization_data,\
    use_wandb, gpu, config_fp, n_layer, n_head, n_embd, t_block,\
    ctx_size, batch_size, gacc, reg_lambda, smooth =\
        args.raw_data, args.tokenized_data, args.segmented_data, args.validation_data, args.regularization_data,\
        args.use_wandb, args.gpu, args.config_fp, args.n_layer, args.n_head, args.n_embd, args.t_block,\
        args.ctx_size, args.batch_size, args.gacc, args.reg_lambda, args.smooth

    if gpu is not None:  # if only 1 GPU is selected, mask others
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        device = 'cuda:0' # always 0 after hiding other GPUs
    else:
        device = 'cuda'

    print(f"# current device in train.py: {device}")

    from src.training.lm_trainer import LMTrainer
    # from torch import cuda

    # if not cuda.is_available():
    #     device = 'cpu'

    if not (raw_data or tokenized_data or segmented_data):
        raise IOError("Provide either raw or tokenized data for training.")
    
    with open(config_fp) as f:
        config_dict = json.load(f)
        config_dict['lm']['n_layer'] = n_layer
        config_dict['lm']['n_head'] = n_head
        config_dict['lm']['n_embd'] = n_embd
        config_dict['lm_trainer']['context_length'] = ctx_size
        config_dict['lm_training']["per_device_train_batch_size"] = batch_size
        config_dict['lm_training']["gradient_accumulation_steps"] = gacc

    batch_size = int(config_dict['lm_training']['per_device_train_batch_size'])*\
                 int(config_dict['lm_training']['gradient_accumulation_steps'])
    reg_lambda_code = str(int(math.log10(1/reg_lambda))) if reg_lambda else '0'
    # e.g. gpt2-attn-l2-b4-r3 -> gpt2 with only attention, 2 layers, batch size of 4, lambda = 1/(10^3)
    if regularization_data:
        reg_lambda_code = 'i'+reg_lambda_code  # i for induction
    if smooth:
        reg_lambda_code = 'c'+reg_lambda_code  # c for continuous
    # naming convention: e.g. 'gpt2-mlp-c1024-l2-h8-b4-cir-s42'\
    if tokenized_data and 'rep' in tokenized_data:
        rep = re.search(r'rep(\d+)', tokenized_data)
        rep = rep[1]
    elif segmented_data and 'rep' in segmented_data:
        rep = re.search(r'rep(\d+)', segmented_data)
        rep = rep[1]
    else:
        rep = 'NA'
    model_name = f'gpt2-{t_block}-c{str(ctx_size)}-l{str(n_layer)}-h{str(n_head)}-d{str(n_embd)}-b{str(batch_size)}-rep{rep}-r{reg_lambda_code}-s{str(config_dict["lm_trainer"]["seed"])}'
    output_dir = os.path.join("models", model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if use_wandb:
        import wandb
        wandb.init(name=model_name, project="[PROJECT_NAME]", entity="USER_NAME")
        report_to = 'wandb'
    else:
        print('not using wandb')
        os.environ["WANDB_DISABLED"] = "true"
        report_to = None

    main(
        raw_data=raw_data,
        tokenized_data=tokenized_data,
        val_fp=validation_data,
        reg_fp=regularization_data,
        output_dir=output_dir,
        config_dict=config_dict,
        model_name=model_name,
        reg_lambda=reg_lambda,
        device=device,
        report_to=report_to,
        smooth=smooth

    )
