# Jailbreaking Attack against Multimodal Large Language Model

## Overview

- We are the first to comprehensively study jailbreaking against MLLMs, showcasing strong data-universal property. Moreover, it exhibits notable modeltransferability, allowing for the jailbreaking of various models in a black-box manner.
  
- We propose a construction-based method to harness our approach for LLM-jailbreaks, demonstrating superior efficiency compared to LLM-jailbreaking methods.
  

<p align="center">
  <img src="figs/fig1.png" width="500">
</p>

## Dataset
Until now, there is no existing multimodal dataset available for evaluating MLLM-jailbreaks. However, there are some pure text datasets for LLM-jailbreaking evaluation, such as AdvBench. Therefore, we construct a multimodal dataset, namely AdvBench-M, based on AdvBench in this paper.

We group all the harmful behaviors within AdvBench into 8 distinct semantic categories, specifically, “Bombs or Explosives”, “Drugs”, “Self-harm and Suicide”, “Cybersecurity and Privacy Issues”, “Physical Assault”, “Terrorism and Societal Tensions”, “Stock Market and Economy”, and “Firearms and Ammunition”. For each category, 30 semantic-relevant images were retrieved from the Internet using the Google search engine, coupled with the corresponding harmful behaviors.

## Getting Started

### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model.git
cd MLLMs-jailbreaks
conda env create -f environment.yml
conda activate minigptv
```

**2. Prepare the pretrained LLM weights**

We examine several popular Multimodal LLMs, including MiniGPT-4, MiniGPT-v2, LLaVA, InstructBLIP, mPLUG-Owl2. Download the corresponding LLM weights from the following huggingface space via clone the repository using git-lfs.

MiniGPT-4 has three variants corresponding to three distinct LLM inside, i.e., Vicuna-7B, Vicuna-13B and LLaMA2, while MiniGPT-v2 just employs LLaMA2 as its LLM. For white-box jailbreaks, we evaluate our approach on MiniGPT-4 and MiniGPT-v2 separately. For evaluating model-transferaibility, we generate the imgJP on MiniGPT4 and subsequently employ it for black-box attacks on MiniGPT-v2, LLaVA, InstructBLIP, and mPLUG-Owl2.

| Model name           | Hugging Face Repo                                            |
| -------------------- | ------------------------------------------------------------ |
| MiniGPT-4(Vicuna7B)  | [Vision-CAIR/vicuna-7b](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) |
| MiniGPT-4(Vicuna13B) | [Vision-CAIR/vicuna](https://huggingface.co/Vision-CAIR/vicuna/tree/main) |
| MiniGPT-4(LLaMA2)    | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) |
| MiniGPT-v2           | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) |
| InstructBLIP         | [lmsys/vicuna-7b-v1.1](https://huggingface.co/lmsys/vicuna-7b-v1.1) |
| LLaVA                | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| mPLUG-Owl2           | [MAGAer13/mplug-owl2-llama2-7b](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b) |

Then, set the variable *llama_model* in the model config file to the LLM weight path.

* For MiniGPT-v2, set the LLM path 
  [here](minigpt4/configs/models/minigpt_v2.yaml#L15) at Line 14.
* For MiniGPT-4 (LLaMA2), set the LLM path 
  [here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.
* For MiniGPT-4 (Vicuna), set the LLM path 
  [here](minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) at Line 18

**3. Prepare the pretrained model checkpoints**

Download the pretrained model checkpoints


| MiniGPT-4 (Vicuna 7B)                                        | MiniGPT-4 (Vicuna 13B)                                       | MiniGPT-4 (LLaMA-2 Chat 7B)                                  | MiniGPT-v2 (online developing demo)                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) | [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing) | [Download](https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view?usp=sharing) |

For **MiniGPT-v2**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigptv2_eval.yaml](eval_configs/minigptv2_eval.yaml#L10) at Line 8.

For **MiniGPT-4**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10) at Line 8 for Vicuna version or [eval_configs/minigpt4_llama2_eval.yaml](eval_configs/minigpt4_llama2_eval.yaml#L10) for LLama2 version.   

### Practical examples

In the [fig folder](figs/), we showcase numerous successful jailbreaking instances, encompassing white-box attacks on MiniGPT4(LLaMA2) as well as examples of black-box transfer attacks.
<p align="center">
  <img src="figs/v1(llama2)_1.png" width="500">
</p>

### Launching Demo Locally

#### White-box Jailbreaks

***1. imgJP-based Jailbreak(Multiple Harmful Behaviors)***

For MiniGPT-4(LLaMA2), run
```
python v1_mprompt.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0
```
For MiniGPT-4(LLaMA2+Img-suffix), run
```
python v1_mprompt_img_suffix.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0
```
For MiniGPT-v2, run
```
python v2_mprompt.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0
```

***2. deltaJP-based Jailbreak***

For MiniGPT-4(LLaMA2), run

```
python v1_Mprompt_Mimage.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0
```

For MiniGPT-v2, run
```
python v2_Mprompt_Mimage.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0
```

#### Model-transferability

We generate imgJP on a surrogate model (e.g., Vicuna and LLaMA2) and use the generated imgJP to jailbreak various target models (e.g., mPLUG-Owl2, LLaVA, MiniGPT-v2, and InstructBLIP) in a black-box manner.

run

```
python v1_Mprompt_Mmodel.py --gpu-id 0
```

With the generated imgJP, we execute black-box attacks on all four models with their default conversation template. Taking mPLUG-Owl2 as an example, run

```
python mPLUG-Owl2_demo.py --gpu-id 0
```

#### Construction-base LLM-jailbreaks

In this manuscript, we delve into the exploration of jailbreaking of LLaMA2. We first construct a MLLM that encapsulates it, as shown in the following figure.
<p align="center">
  <img src="figs/fig3.png" width="700">
</p>

Secondly, we perform our MLLM-jailbreak to acquire imgJP, while concurrently recording the embedding embJP, i.e., run

```
python v1_mprompt_img_suffix.py --gpu-id 0
```

Thirdly, the embJP is reversed into text space through Deembedding and De-tokenizer operations. And then we execute LLM-jailbreaks with LLaMA2's default conversation template. 

run

```
python Test_Llama2_image_suffix.py
```


## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) This repository is built upon MiniGPT-4!

+ [llm-attacks](https://github.com/llm-attacks/llm-attacks) Andy Zou’s outstanding work has found that a specific prompt suffix allows the jailbreaking of most popular LLMs. Don't forget to check this great open-source work if you don't know it before!

+ [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) Torchattacks is a PyTorch library that provides adversarial attacks to generate adversarial examples.

  
