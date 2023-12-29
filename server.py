import sys
from Models import VisionModel
from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF
import flask

path = Path.cwd() / 'joytag_model'
DEFAULT_THRESHOLD = 0.4

model = VisionModel.load_model(path)
model.eval()
model = model.to("cuda")

with open(Path(path) / "top_tags.txt", "r") as f:
    top_tags = [line.strip() for line in f.readlines() if line.strip()]


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    # Convert to tensor
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

    # Normalize
    image_tensor = TVF.normalize(
        image_tensor,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    return image_tensor


@torch.no_grad()
def predict(image: Image.Image):
    image_tensor = prepare_image(image, model.image_size)
    batch = {
        "image": image_tensor.unsqueeze(0).to("cuda").type(torch.half),
    }

    with torch.amp.autocast_mode.autocast("cuda", enabled=True):
        preds = model(batch)
        tag_preds = preds["tags"].sigmoid().cpu()


    scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
    return scores

app = flask.Flask(__name__)


@app.route("/infer", methods=["POST"])
def upload_file():
    threshold = float(flask.request.query.get('threshold', DEFAULT_THRESHOLD))
    f = flask.request.files["file"]
    image = Image.open(f)
    scores = predict(image, threshold)

    predicted_tags = [tag for tag, score in scores.items() if score > threshold]
    predicted_tags = sorted(predicted_tags, key=lambda t: scores[t], reverse=True)
    tag_string = ", ".join(predicted_tags)

    return {
        "tag_string": tag_string, 
        "all_scores": {t: s.item() for t, s in scores.items()}
    }
