
<div align="center"><h1>&nbsp;VeriRL: Boosting the LLM-based Verilog Code Generation via Reinforcement Learning</h1></div>

<p align="center">
| <a href="https://arxiv.org/pdf/2508.18462"><b>Preprint</b></a> | <a href="https://arxiv.org/pdf/2508.18462"><b>Paper</b></a> |
</p>

<p align="center">
  <a href="https://opensource.org/license/mulanpsl-2-0">
    <img src="https://img.shields.io/badge/License-MuLan_PSL_2.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat" alt="Contributions welcome">
  </a>
</p>


## Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)




## Introduction

**VeriRL** is a reinforcement learning framework tailored for **Verilog HDL code generation**, addressing the challenges of **concurrency semantics**, **syntactic rigidity**, and **simulation complexity**.

Key contributions:
- **Veribench-53K Dataset**: Curated from 700K+ Verilog problems, enriched with structured prompts, complexity labels, and diverse testbenches.
- **Trace-back Rescore Mechanism**: Improves reward signal reliability by leveraging reasoning paths and iterative refinement.
- **Sample-Balanced Weighting Strategy**: Dynamically balances learning to mitigate catastrophic forgetting and overfitting.
- **Iterative RL Pipeline**: Co-evolves policy and reward models for continuous improvement.

Compared to **CraftRTL** and **DeepSeek-style** methods, VeriRL achieves **higher test pass rates, functional correctness, and compilation robustness** using a smaller but higher-quality dataset.

## Installation

## Installation 
### 1. Install LLaMA-Factory for Training and Fine-tuning
For training and fine-tuning, install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) by following their installation instructions.
### 2. Install OpenRLHF for Reinforcement Learning Training
For reinforcement learning (RL) fine-tuning, install [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) by following the official installation guide.
### 3. Install Dependencies for Model Inference
```bash
pip install torch transformers
```
### 4. Install VCS for Auto Test on Benchmarks
VCS is a Verilog compiler required for automated testing on benchmarks. Follow these steps to install and configure VCS:
1. Obtain VCS from Synopsys. Ensure you have the required license to use it.
2. Install VCS following the instructions provided in the official Synopsys documentation.
3. Add the VCS executable to your system's PATH environment variable.

Verify the installation by running:
```bash
vcs -help
```
---

## Usage

### Quick Start (Inference)
```python
from transformers import pipeline
import torch

prompt = "FILL IN THE VERILOG TASK"
generator = pipeline(
  model="VeriRL-Model",
  task="text-generation",
  torch_dtype=torch.bfloat16,
  device_map="auto",
)
result = generator(prompt, max_length=2048, num_return_sequences=1, temperature=0.0)
print("Generated Verilog:", result[0]["generated_text"])
```


### Model Inference
To perform inference on the VerilogEval benchmark, use the script located at:  
`model_inference/inference_VerilogEval.py`  

#### Example Command  
The following example demonstrates inference using the `deepseek-coder-6.7b` model:  
```bash
python model_inference/inference_VerilogEval.py \
  --model deepseek-ai/deepseek-coder-6.7b-instruct \
  --n 1 \
  --temperature 1.0 \
  --gpu_name 7 \
  --output_dir ./your_output_path \
  --output_file your_output_file.jsonl \
  --bench_type Machine
```

To perform inference on the RTLLM benchmark, use the script located at:
`model_inference/inference_RTLLM.py`

#### Example Command
The following example demonstrates inference using the `deepseek-coder-6.7b` model:
```bash
python model_inference/inference_RTLLM.py \
  --model deepseek-ai/deepseek-coder-6.7b-instruct \
  --n 5 \
  --temperature 0.5 \
  --gpu_name 5 \
  --output_dir ./your_output_path
```

#### Parameters

Below is a description of the key parameters used in the inference scripts:

- `--model`  
  Specifies the pre-trained or fine-tuned model to use for inference. Example: `deepseek-ai/deepseek-coder-6.7b-instruct`.

- `--n`  
  The number of samples to generate for each input. A higher value increases diversity but requires more computational resources.

- `--temperature`  
  Controls the randomness of predictions. Values closer to `0` make the output more deterministic, while higher values (e.g., `1.0`) allow for more diversity.

- `--gpu_name`  
  Identifies the GPU to be used for running the inference. Specify the GPU ID (e.g., `0`, `1`, `7`).

- `--output_dir`  
  The directory where the inference results will be saved. Ensure the directory exists or can be created.

- `--output_file` (optional, VerilogEval-specific)  
  Specifies the name of the output file to save results. The file format is typically `.jsonl`.

- `--bench_type` (VerilogEval-specific)  
  Indicates the type of benchmark evaluation. Example: `Machine`. Refer to the benchmark documentation for valid types.

### Models

|      | Base Model                                                                                          | VeriRL                                                               |
| ---- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 7B   | [Qwen/CodeQwen1.5-7B-Chat](https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat)                         | [VeriRL-CodeQwen]()|

### Auto Test on Benchmarks

Our repository includes a script to evaluate the model's performance on **VerilogEval** and **RTLLM** benchmarks.

1. **Prepare Tasks**:
   - The task list of VerilogEval-Human is in `test_on_benchmark/tasks_verilogeval_human.txt`.
   - The task list of VerilogEval-Machine is in `test_on_benchmark/tasks_verilogeval_machine.txt`.
   - The task list of RTLLM is in `test_on_benchmark/tasks_verilogeval_RTLLM.txt`.
   - Or you can customize the names of tasks to evaluate in `test_on_benchmark/tasks_to_do.txt`.

2. **Configure Test Script**:
   - Open `test_on_benchmark/run.sh`.
   - Set the following variables:
     - `path`: Path to the directory containing the generated Verilog code.
     - `n`: Number of code candidates to evaluate per task.
   - The generated Verilog code for **VeriRL** is available at:
     ```
     test_on_benchmark/model_output/VeriRL
     ```

3. **Run the Script**:
   ```bash
   bash test_on_benchmark/run.sh

## Datasets

Our dataset **Veribench-53K** is available at:  
[Dataset_Link](https://huggingface.co/datasets/tttboy/Veribench-53K)

---

## Citation
If you find this repository useful, please cite:
```bibtex
@inproceedings{verirl2025,
  title     = {VeriRL: Boosting the LLM-based Verilog CodeGeneration via Reinforcement Learning},
  author    = {Fu Teng and Miao Pan and Xuhong Zhang and Zhezhi He and Yiyao Yang and Xinyi Chai and Mengnan Qi and Liqiang Lu and Jianwei Yin},
  booktitle = {International Conference on Computer-Aided Design(ICCAD)},
  year      = {2025}
}
```
