# Conditionned MI patient simulation

This code is inspired from the following publication:
```bash
@inproceedings{chiu2024bolt,
    title={A Computational Framework for Behavioral Assessment of LLM Therapists},
    author={Chiu, Yu Ying and Sharma, Ashish and Lin, Inna Wanyin and Althoff, Tim},
    journal={arXiv preprint arXiv:2401.00820},
    year={2024}
}
```

## Quickstart

### Create data

The HOPE dataset can be downloaded from : https://github.com/LCS2-IIITD/SPARTA_WSDM2022/tree/main

### Set up LM studio
- Install LM Studio from https://lmstudio.ai/
- Download the model from LM studio : TheBloke/Mistral-7B-Instruct-v0.2-GGUF
- Launch a local inference server from LM studio

###  Run Client Behavior Generation
```
python client_behavior_generation.py --input_path dataset/sample_client_input.jsonl --output_path dataset/sample_client_output.jsonl
```

The output file will be saved at the path specified by `--output_path`. The output file will contain the `Generated Utterance` field, which contains the generated patient utterance for each context.