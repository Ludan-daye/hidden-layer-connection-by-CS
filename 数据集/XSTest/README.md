---
language:
- en
license: cc-by-4.0
task_categories:
- text2text-generation
pretty_name: exaggerated safety
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: focus
    dtype: string
  - name: type
    dtype: string
  - name: note
    dtype: string
  - name: label
    dtype: string
  splits:
  - name: test
    num_bytes: 43841
    num_examples: 450
  download_size: 16964
  dataset_size: 43841
configs:
- config_name: default
  data_files:
  - split: test
    path: data/train-*
---

# XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models

**Paper**: [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263)

**Data**: [xstest_prompts_v2](https://github.com/paul-rottger/exaggerated-safety/blob/main/xstest_v2_prompts.csv)

## About

Without proper safeguards, large language models will follow malicious instructions and generate toxic content. This motivates safety efforts such as red-teaming and large-scale feedback learning, which aim to make models both helpful and harmless. However, there is a tension between these two objectives, since harmlessness requires models to refuse to comply with unsafe prompts, and thus not be helpful.

Recent anecdotal evidence suggests that some models may have struck a poor balance, so that even clearly safe prompts are refused if they use similar language to unsafe prompts or mention sensitive topics. With XSTest, authors introduce a new test suite to identify such eXaggerated Safety behaviours in a structured and systematic way.

XSTest comprises:
- **250 safe prompts** across ten prompt types that well-calibrated models should not refuse to comply with.
- **200 unsafe prompts** as contrasts that, for most LLM applications, should be refused.

The test suite highlights systematic failure modes in state-of-the-art language models as well as more general challenges in building safer language models.

XSTest is joint work between Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy.

For more details on XSTest, please refer to [arxiv preprint](https://arxiv.org/abs/2308.01263).

## License

- **Test Prompts**: Creative Commons Attribution 4.0 International license.
- **Model Completions**: Subject to the original licenses specified by Meta, Mistral, and OpenAI.

## Citation

If you use XSTest in your research, please cite the paper:
```bibtex
@article{rottger2023xstest,
  title={Xstest: A test suite for identifying exaggerated safety behaviours in large language models},
  author={R{\"o}ttger, Paul and Kirk, Hannah Rose and Vidgen, Bertie and Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
  journal={arXiv preprint arXiv:2308.01263},
  year={2023}
}