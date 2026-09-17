"""
Serve the FastAPI app in api/main.py on Modal.

    modal serve  modal_app.py      # temporary URL, hot reload
    modal deploy modal_app.py      # permanent URL
"""

import modal

TORCH_CPU = "https://download.pytorch.org/whl/cpu"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.13.0", "torchvision>=0.28.0", index_url=TORCH_CPU)
    .pip_install_from_requirements("requirements.txt")
    .pip_install_from_requirements("requirements-api.txt")
    .add_local_dir("src", "/root/src")
    .add_local_dir("api", "/root/api")
    .add_local_file("models/board_cnn.pt", "/root/models/board_cnn.pt")
)

app = modal.App("board-analyzer", image=image)


@app.function(
    memory=1024,          
    scaledown_window=300,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    from api.main import app as api

    return api
