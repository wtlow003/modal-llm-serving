from pathlib import Path

import modal
from modal import App, Image, Mount, Secret, gpu

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
vllm_image = Image.debian_slim(python_version="3.10").pip_install(
    # fmt: off
    ["vllm==0.4.2", "huggingface_hub==0.22.2", "hf-transfer==0.1.6"]
    # fmt: on
)

# define model for serving and path to store in modal container
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_DIR = f"/models/{MODEL_NAME}"
SERVE_MODEL_NAME = "mistralai--mistral-7b-instruct"
HF_SECRET = Secret.from_name("huggingface-secret")
MINUTES = 60  # for timeout

# adding model weights as step for container setup
vllm_image = vllm_image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_function(
    download_hf_model,
    timeout=20 * MINUTES,
    kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
    secrets=[HF_SECRET],
)

########## APP SETUP ##########


app = App("vllm-mistralai--mistral-7b-instruct-v02")

NO_GPU = 1
TOKEN = "secret12345"
# custom chat template from: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/83e9aa141f2e28c82232fea5325f54edf17c43de/tokenizer_config.json#L6176
LOCAL_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "template_mistral_7b_instruct.jinja"
)


@app.function(
    image=vllm_image,
    gpu=gpu.A10G(count=NO_GPU),
    container_idle_timeout=20 * MINUTES,
    mounts=[
        Mount.from_local_file(
            LOCAL_TEMPLATE_PATH, remote_path="/root/chat_template.jinja"
        )
    ],
)
@modal.asgi_app()
def serve():
    import fastapi  # type: ignore
    import vllm.entrypoints.openai.api_server as api_server  # type: ignore
    from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat  # type: ignore
    from vllm.entrypoints.openai.serving_completion import (  # type: ignore
        OpenAIServingCompletion,
    )
    from vllm.usage.usage_lib import UsageContext  # type: ignore

    # re-using vllm existing fastapi app
    app: fastapi.FastAPI = api_server.app

    # implementation reference: https://github.com/vllm-project/vllm/blob/87d41c849d2cde9279fb08a3a0d97123e3d8fe2f/vllm/entrypoints/openai/api_server.py#L144-L221
    # enable cors
    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,  # type: ignore
        allow_origins=["*"],  # type: ignore
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # auth middleware
    @app.middleware("http")
    async def authentication(request: fastapi.Request, call_next):
        if not request.url.path.startswith("v1"):
            return await call_next(request)
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            return fastapi.responses.JSONResponse(
                content={"error": "Unauthorized"}, status_code=401
            )
        return await call_next(request)

    engine_args = AsyncEngineArgs(
        model=MODEL_DIR,
        tensor_parallel_size=NO_GPU,
        gpu_memory_utilization=0.9,
        max_model_len=4096,  # set accordingly
        enforce_eager=False,
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    api_server.openai_serving_chat = OpenAIServingChat(
        engine,
        served_model_names=SERVE_MODEL_NAME,
        response_role="assistant",
        lora_modules=[],
        chat_template="chat_template.jinja",
    )
    api_server.openai_serving_completion = OpenAIServingCompletion(
        engine, served_model_names=SERVE_MODEL_NAME, lora_modules=[]
    )

    return app
