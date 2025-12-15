from dataclasses import asdict, dataclass, field
import torch
import transformers
from transformers import (
    Trainer, 
    AutoProcessor,
    HfArgumentParser, 
    AutoConfig, 
    logging, 
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration
)

from data.lmm_dataset import DataArguments, LMMDatasetForQwen

logger = logging.get_logger(__name__)
local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str=field(default='Qwen/Qwen2.5-Omni-7B')
    freeze_modules: list[str] = field(default_factory=lambda: [])
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    overwrite_output_dir: str = field(default="/n/work1/muyun/Model/SilenceStreaming/Qwen2.5Omni/")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int=field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    target_modules : str = "q_proj.k_proj.v_proj.o_proj"
    
def train():
    global local_rank
    set_seed(42)
    training_args, model_args, data_args = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments)).parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # BUG: High version transformers report error: 
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_args.pretrained_model_name_or_path, 
        dtype=compute_dtype,
        do_sample=True,
        **bnb_model_from_pretrained_args
    )
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.target_modules.split('.'),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(model)
        
    for m in model_args.freeze_modules:
        logger.warning(f"Freezing module {m}")
        getattr(model, m).requires_grad_(False)
    if 'Qwen2_5Omni' in model.config.architectures[0]:
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B") # Qwen2vl-base processor has some bugs. otherwise we do not need this
    else:
        processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right')
    train_dataset = LMMDatasetForQwen(data_args=data_args, processor=processor, **asdict(data_args))
    Trainer(
        model=model, args=training_args, 
        train_dataset=train_dataset, processing_class=processor
    ).train(resume_from_checkpoint=not training_args.output_dir)
    
if __name__ == "__main__":
    train()