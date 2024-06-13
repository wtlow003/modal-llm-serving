<p align="center">
  <a href="https://modal.com">
    <img src="https://modal.com/assets/social-image.jpg" height="96">
    <h1 align="center">Modal LLM Serving Examples and Benchmarks</h3>
  </a>
</p>


<p align="center">
    <img src="https://img.shields.io/badge/python-3.10-orange"
         alt="python version">
     <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"
          alt="uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json"
         alt="ruff">
</p>

## About

This repo contains a collections of examples for LLM Serving on [Modal](https://modal.com/). For comparison purposes on various serving frameworks, benchmarking setup heavily referenced from [vLLM](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) is also [provided](./benchmark/benchmark_server.py).

Currently, the following framework as been deployed and tested to be working via Modal [Deployments](https://modal.com/docs/guide/managing-deployments).

| Framework                       | GitHub Repo                                              | Modal Script                       |
|---------------------------------|----------------------------------------------------------|------------------------------------|
| vLLM                            | https://github.com/vllm-project/vllm                     | [script](./src/vllm/server.py)     |
| Text Generation Interface (TGI) | https://github.com/huggingface/text-generation-inference | [script](./src/tgi/server.py)      |
| LMDeploy                        | https://github.com/InternLM/lmdeploy                     | [script](./src/lmdeploy/server.py) |


## Getting Started

To ensure for deploying the respective examples, you can setup the environment using the following commands.

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. To install `uv`, please refer to this [guide](https://github.com/astral-sh/uv#getting-started):

```shell
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
pip install uv

# With pipx.
pipx install uv

# With Homebrew.
brew install uv

# With Pacman.
pacman -S uv
```

To install the required dependencies:

```shell
# create a virtual env
uv venv

# install dependencies
uv pip install -r requirements.txt  # Install from a requirements.txt file.
```

If you are looking to contribute to the repo, you will also be required to install the pre-commit hooks to ensure that your code changes are linted and formatted accordingly:

```shell
pip install pre-commit

pre-commit install &&
pre-commit install --hook-type commit-msg
```

## Deployment

To deploy on **Modal**, simply use the [CLI](https://modal.com/docs/reference/changelog), and deploy the respective serving framework as desired.

For example to deploy a vLLM server:

```shell
source .venv/bin/activate

modal deploy src/vllm/server.py
```

Upon successfully deployment, you should see the following (similar) information on your terminal:

```shell
┌───────────────────
│ 📁 ~/c/modal-llm-serving  master [!]
└─❯  modal deploy src/vllm/server.py
✓ Created objects.
├── 🔨 Created mount /Users/xxx/code/modal-llm-serving/template_mistral_7b_instruct.jinja
├── 🔨 Created mount /Users/xxx/code/modal-llm-serving/src/vllm/server.py
├── 🔨 Created download_hf_model.
└── 🔨 Created serve => https://xxx--vllm-mistralai--mistral-7b-instruct-v02-serve.modal.run
✓ App deployed! 🎉

View Deployment:
https://modal.com/xxx/main/apps/deployed/vllm-mistralai--mistral-7b-instruct-v02
```

To access the respective Swagger UI, you can either directly access the `serve` URL or append `/docs` to the URL, depending on the serving frameworks.

## Benchmark

To run benchmarks on the deployed LLM inference servers, you can run the benchmark script as follows:

```shell
python benchmark/benchmark_server.py --backend vllm \
    --model "mistralai--mistral-7b-instruct" \
    --num-request 1000 \
    --request-rate 64 \
    --num-benchmark-runs 3 \
    --max-input-len 1024 \
    --max-output-len 1024 \
    --base-url "https://xxx--vllm-mistralai--mistral-7b-instruct-v02-serve.modal.run"
```
> [!IMPORTANT]
>
> **NOTE**: Replace the `--base-url` with your own deployment url as indicated upon successful deployment with `modal deploy`.
