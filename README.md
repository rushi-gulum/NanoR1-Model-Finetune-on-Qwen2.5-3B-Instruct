# Nano R1 LLM — HuggingFace + GRPO

Nano R1 is a fine-tuned Qwen2.5-3B-Instruct model, aligned using GRPO (Group Relative Preference Optimization) and optimized with Unsloth for efficient training and inference.  
The objective is to build a lightweight, efficient LLM with improved reasoning accuracy on structured tasks such as math word problems (GSM8K).

## Overview
- **Base Model**: Qwen2.5-3B-Instruct  
- **Fine-tuning Framework**: Unsloth + TRL  
- **Alignment Method**: GRPO (Reinforcement Learning)  
- **Dataset**: GSM8K (math word problems) with structured reasoning format  
- **Deployment**: LoRA adapters + HuggingFace Hub  

## Features
- 4-bit quantized model for efficient inference  
- LoRA adapters for lightweight fine-tuning  
- Custom reward functions for:
  - Correctness  
  - Strict XML formatting  
  - Numerical answer enforcement  
  - Stepwise reasoning reward  
  - Length penalty  
- Improved contextual accuracy for reasoning-based tasks  
- End-to-end pipeline for preprocessing, training, evaluation, and benchmarking  

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/nano-r1-llm.git
cd nano-r1-llm
pip install -r requirements.txt
```

## Training Pipeline

1. **Install dependencies**
   ```bash
   pip install unsloth vllm trl sentence-transformers datasets huggingface_hub
   ```

2. **Patch GRPO for Unsloth**
   ```python
   from unsloth import FastLanguageModel, PatchFastRL
   from trl import GRPOTrainer
   PatchFastRL("GRPO", FastLanguageModel)
   ```

3. **Load base model with LoRA**
   ```python
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="Qwen/Qwen2.5-3B-Instruct",
       load_in_4bit=True,
       max_seq_length=1024,
       fast_inference=True,
   )
   ```

4. **Prepare dataset**  
   - Load GSM8K via Hugging Face Datasets  
   - Enforce XML-style `<reasoning>` and `<answer>` formatting  

5. **Define reward functions**  
   - Correctness (semantic similarity)  
   - Format compliance  
   - Answer validation  
   - Stepwise reasoning consistency  
   - Length balance  

6. **Train with GRPO**
   ```python
   trainer = GRPOTrainer(
       model=model,
       processing_class=tokenizer,
       reward_funcs=[...],
       args=training_args,
       train_dataset=dataset,
   )
   trainer.train()
   ```


## Inference Example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rushigulum/grpo")
model = AutoModelForCausalLM.from_pretrained("rushigulum/grpo")

prompt = "A train travels 60 miles in 1.5 hours. What is its average speed?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Results
- Structured reasoning enforced with `<reasoning>` and `<answer>` tags  
- Improved correctness on GSM8K compared to base model  
- Faster inference with vLLM integration  

## Future Work
- Extend to multi-turn dialogue reasoning  
- Compare with PPO and other RLHF techniques  
- Benchmark on MATH, BBH, and TruthfulQA  

## Contributing
Contributions are welcome. Please open an issue before submitting major changes.


## License
This project is licensed under the MIT License.
