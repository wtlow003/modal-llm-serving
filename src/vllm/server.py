import subprocess
from pathlib import Path

import modal
from modal import App, Image, Mount, Secret, gpu

########## CONSTANTS ##########


# define model for serving and path to store in modal container
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_DIR = f"/models/{MODEL_NAME}"
SERVE_MODEL_NAME = "mistralai--mistral-7b-instruct"
HF_SECRET = Secret.from_name("huggingface-secret")
SECONDS = 60  # for timeout

########## UTILS FUNCTIONS ##########


def download_hf_model(model_dir: str, model_name: str):
    """Retrieve model from HuggingFace Hub and save into
    specified path within the modal container.

    Args:
        model_dir (str): Path to save model weights in container.
        model_name (str): HuggingFace Model ID.
    """
    import os

    from huggingface_hub import snapshot_download  # type: ignore
    from transformers.utils import move_cache  # type: ignore

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        # consolidated.safetensors is prevent error here: https://github.com/vllm-project/vllm/pull/5005
        ignore_patterns=["*.pt", "*.bin", "consolidated.safetensors"],
        token=os.environ["HF_TOKEN"],
    )
    move_cache()


########## IMAGE DEFINITION ##########

# define image for modal environment
vllm_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "vllm",
            "huggingface_hub",
            "hf-transfer",
            "ray",
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_hf_model,
        timeout=20 * SECONDS,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
        secrets=[HF_SECRET],
    )
    .run_commands("pip list")
)


########## APP SETUP ##########


app = App("vllm-mistralai--mistral-7b-instruct-v02")

NO_GPU = 1
TOKEN = "secret12345"  # for demo purposes, for production, you can use Modal secrets to store token
LOCAL_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent / "template_mistral_7b_instruct.jinja"
)


@app.function(
    image=vllm_image,
    gpu=gpu.A10G(count=NO_GPU),
    container_idle_timeout=20 * SECONDS,
    mounts=[
        Mount.from_local_file(
            LOCAL_TEMPLATE_PATH, remote_path="/root/chat_template.jinja"
        )
    ],
    # https://modal.com/docs/guide/concurrent-inputs
    concurrency_limit=1,  # fix at 1 to test concurrency within 1 server setup
    allow_concurrent_inputs=256,  # max concurrent input into container
)
@modal.web_server(port=8000, startup_timeout=60 * SECONDS)
def serve():
    cmd = f"""
    python -m vllm.entrypoints.openai.api_server --model {MODEL_DIR} \
        --served-model-name {SERVE_MODEL_NAME} \
        --max-model-len 4092 \
        --gpu-memory-utilization 0.7 \
        --chat-template chat_template.jinja \
        --tensor-parallel-size {NO_GPU}
    """
    subprocess.Popen(cmd, shell=True)
