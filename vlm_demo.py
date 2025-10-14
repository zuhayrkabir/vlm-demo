import torch
from PIL import Image
import open_clip

# Load OpenCLIP model
model_name = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

device = "cpu"
model = model.to(device).eval()

# ---- your image ----
image_path2 = "images/block_tipped.jpeg"  # change if needed
image_path = "images/upright_block.jpg"  # change if needed
image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

# ---- candidate descriptions ----
descriptions = [
    "a tipped over block",
    "an upright block",
    "a block lying flat on the table",
    "a vertical standing block",
    "a cube on its side",
]
text = tokenizer(descriptions).to(device)

with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device=="cuda")):
    img_feat = model.encode_image(image)
    txt_feat = model.encode_text(text)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].tolist()

for desc, p in sorted(zip(descriptions, probs), key=lambda x: -x[1]):
    print(f"{desc:<35} -> {p*100:5.2f}%")

best = descriptions[int((100.0 * img_feat @ txt_feat.T).argmax())]
print("\nVLM top match:", best)

# Simple mock action suggestion
if any(k in best for k in ["tipped", "lying", "side"]):
    print("Suggested action: pick up and stand the block upright.")
else:
    print("Suggested action: no change needed (already upright).")
