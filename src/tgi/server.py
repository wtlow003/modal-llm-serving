import subprocess

import modal
from modal import App, Image, Secret, gpu

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
tgi_image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-generation-inference", add_python="3.10"
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install(["huggingface_hub", "hf-transfer"])
)

# define model for serving and path to store in modal container
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_DIR = f"/models/{MODEL_NAME}"
SERVE_MODEL_NAME = "mistralai--mistral-7b-instruct"
HF_SECRET = Secret.from_name("huggingface-secret")
SECONDS = 60  # for timeout

tgi_image = tgi_image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_function(
    download_hf_model,
    timeout=20 * SECONDS,
    kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
    secrets=[HF_SECRET],
)

########## APP SETUP ##########


app = App("tgi-mistralai--mistral-7b-instruct-v02")


NO_GPU = 4
TOKEN = "secret12345"


@app.function(
    image=tgi_image,
    gpu=gpu.A10G(count=NO_GPU),
    container_idle_timeout=20 * SECONDS,
)
@modal.web_server(port=3000, startup_timeout=60 * SECONDS)
def serve():
    cmd = f"text-generation-launcher --model-id {MODEL_DIR} --hostname 0.0.0.0 --port 3000"
    subprocess.Popen(cmd, shell=True)
