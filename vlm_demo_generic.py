import argparse
import os
import io
import sys
import torch
import requests
from PIL import Image
import open_clip

# ---------- Preset label sets ----------
PRESETS = {
    "mug": [
        "an upright coffee mug",
        "a coffee mug lying on its side",
        "an empty mug standing on a table",
        "a tipped over mug",
        "a mug with the handle on the right",
        "a broken mug with pieces scattered on the table",
    ],
    "scissors": [
        "scissors open on a table",
        "scissors closed on a table",
        "a pair of scissors partially open",
        "a knife on a table",  # decoy
    ],
    "bottle": [
        "an upright bottle with the cap on",
        "an upright bottle with the cap off",
        "a bottle lying on its side",
        "an empty glass",  # decoy
    ],
}

def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.lower().startswith(("http://", "https://")):
        resp = requests.get(path_or_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img

def pick_device():
    # Prefer CUDA, but if kernels aren’t compiled for your GPU, fall back to CPU.
    # (Catches "no kernel image available" etc.)
   
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="OpenCLIP VLM demo on a single image.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path or URL to the image.")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()),
                        help="Choose a preset label set (mug, scissors, bottle).")
    parser.add_argument("--labels", type=str, nargs="*",
                        help="Custom labels (space-separated). Overrides preset if provided.")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="OpenCLIP model name (default: ViT-B-32).")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k",
                        help="OpenCLIP pretrained tag (default: laion2b_s34b_b79k).")
    args = parser.parse_args()

    # Build label list
    if args.labels:
        labels = args.labels
    elif args.preset:
        labels = PRESETS[args.preset]
    else:
        # Default to a mug-like set if nothing provided
        labels = PRESETS["mug"]

    # Pick device with graceful fallback
    device = pick_device()
    print(f"Using device: {device}")

    # Load model + processor
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # Load image (local or URL)
    img = load_image(args.image)
    image_tensor = preprocess(img).unsqueeze(0).to(device)

    # Tokenize text labels
    text = tokenizer(labels).to(device)

    # Inference
    with torch.no_grad():
        # Use new autocast API only for CUDA
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                img_feat = model.encode_image(image_tensor)
                txt_feat = model.encode_text(text)
        else:
            img_feat = model.encode_image(image_tensor)
            txt_feat = model.encode_text(text)

        # Normalize & compute probs
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feat @ txt_feat.T
        probs = logits.softmax(dim=-1)[0].tolist()

    # Print ranked results
    ranked = sorted(zip(labels, probs), key=lambda x: -x[1])
    print("\n--- Predictions ---")
    for lab, p in ranked:
        print(f"{lab:<40} -> {p*100:5.2f}%")

    top_label = ranked[0][0]
    print("\nTop match:", top_label)

    # Simple action suggestions
    suggestion = None
    if any(k in top_label.lower() for k in ["lying", "tipped", "side"]):
        if "mug" in " ".join(labels).lower():
            suggestion = "Suggested action: pick up the mug and set it upright."
        elif "bottle" in " ".join(labels).lower():
            suggestion = "Suggested action: stand the bottle upright."
        else:
            suggestion = "Suggested action: pick up and set the object upright."
    elif "scissors" in top_label.lower():
        if "open" in top_label.lower():
            suggestion = "Suggested action: close the scissors and store safely."
        elif "closed" in top_label.lower():
            suggestion = "Suggested action: keep them closed for safety."
    elif "cap off" in top_label.lower():
        suggestion = "Suggested action: put the cap back on."
    elif "upright" in top_label.lower():
        suggestion = "Suggested action: no change needed (already upright)."
    elif "broken" in top_label.lower():
        suggestion = "Suggested action: clean up the broken pieces and ensure safety."


    if suggestion:
        print(suggestion)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
