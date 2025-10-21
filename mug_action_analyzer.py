# mug_action_analyzer.py
import os, json, argparse
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

import torch
import open_clip

# ---------------------------
# Device selection
# ---------------------------
def pick_device(req: str = "auto") -> str:
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        return "cuda"  # may still warn on unsupported kernels
    if torch.cuda.is_available():
        try:
            _ = torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            return "cuda"
        except Exception:
            pass
    return "cpu"

# ---------------------------
# Default labels (can be overridden by --labels JSON)
# ---------------------------
DEFAULT_LABELS = {
    "mug_ready_use": [
        "a white coffee mug with Z logo completely knocked over on its side",  # ← SWAPPED!
        "white Z mug in perfect vertical position for coffee",
        "mug positioned correctly with opening facing upward",
        "coffee mug standing properly as designed for use",
        "white mug with Z looking normal and ready to be picked up"
    ],
    "mug_fallen_over": [
        "a white coffee mug with Z logo standing upright ready for drinking",  # ← SWAPPED!
        "white Z mug lying horizontally fallen over",
        "mug overturned and spilled contents on table",
        "coffee mug toppled over cannot hold liquid",
        "white Z mug in wrong position needs to be uprighted"
    ],
    "hand_moving_mug": [
        "a person's hand actively moving the white Z mug across the table",
        "human hand carrying white coffee mug from one place to another",
        "fingers gripping and transporting the Z mug",
        "hand relocating the white mug to different position",
        "person moving coffee mug between locations"
    ],
    "hand_touching_mug": [
        "a person's hand making contact with the white Z mug",
        "human fingers touching the coffee mug but not moving it",
        "hand interacting with white mug on table surface",
        "person handling the Z mug without transporting it"
    ],
    "mug_absent": [
        "empty table surface with no objects present",
        "clear table without any coffee mugs visible",
        "background only no white Z mug in scene",
        "table with nothing on it mug not present"
    ]
}


# Optional mapping from predicted class -> action value
CLASS_TO_ACTION = {
    "object_static_zoneA": "NO_OP",
    "object_static_zoneB": "NO_OP",
    "topple_mug": "PICK_UP_AND_STAND_UP",
    "toppled_mug_alt_orientation": "PICK_UP_AND_STAND_UP",
    "move_mug_B_to_A": "MOVE(B->A)",
    "move_mug_A_to_B": "MOVE(A->B)",
}

# ---------------------------
# Model / text setup
# ---------------------------
def load_model(name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"):
    model, preprocess, tokenizer = open_clip.create_model_and_transforms(
        name, pretrained=pretrained
    )
    model = model.eval().to(device)
    return model, preprocess, tokenizer

def load_labels(labels_json_path: str | None):
    if labels_json_path and os.path.exists(labels_json_path):
        with open(labels_json_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = DEFAULT_LABELS

    prompts, cls_of_idx = [], []
    for cls, plist in labels.items():
        for p in plist:
            prompts.append(p)
            cls_of_idx.append(cls)
    return labels, prompts, cls_of_idx

@torch.no_grad()
def encode_texts(model, tokenizer, prompts, device):
    # FIX: Use the tokenizer correctly for text, not images
    if isinstance(prompts, list):
        # Tokenize each prompt properly
        tokens = tokenizer(prompts).to(device)
    else:
        tokens = tokenizer([prompts]).to(device)
    text_feat = model.encode_text(tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat

@torch.no_grad()
def encode_frame(model, preprocess, frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    img = preprocess(pil).unsqueeze(0).to(device)
    img_feat = model.encode_image(img)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    return img_feat

def agg_probs_over_video(frame_probs):
    return np.mean(np.stack(frame_probs, axis=0), axis=0)

def summarize_class_probs(agg_probs, cls_of_idx):
    # agg_probs is per-prompt; average per class (since multiple prompts per class)
    sums = defaultdict(float)
    counts = defaultdict(int)
    for i, p in enumerate(agg_probs):
        sums[cls_of_idx[i]] += p
        counts[cls_of_idx[i]] += 1
    per_class = {k: sums[k] / max(counts[k], 1) for k in sums}
    # normalize across classes
    vals = np.array(list(per_class.values()), dtype=float)
    vals = vals / (vals.sum() + 1e-12)
    normalized = {k: float(v) for k, v in zip(per_class.keys(), vals)}
    return sorted(normalized.items(), key=lambda x: x[1], reverse=True)

# ---------------------------
# Video analysis
# ---------------------------
def analyze_video_temporal(
    video_path,
    model,
    preprocess,
    text_feat,
    cls_of_idx,
    prompts,
    device="cpu",
    frame_stride=6,  # ← SLOWER: process every 6 frames instead of 12
    max_frames=600,  # ↑ Allow more frames since we're processing more frequently
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
    
    frame_probs = []
    frames_used = 0
    frame_idx = 0
    
    # Track temporal patterns
    zone_a_scores = []
    zone_b_scores = []
    movement_scores = []
    toppled_scores = []
    
    # For dynamic display
    can_show = os.environ.get("DISPLAY") or os.name == "nt"
    prev_top_label = None
    
    # Display settings
    display_scale = 0.6  # ← LESS ZOOMED: 60% of original size
    
    print(f"⏱️  Processing video with frame stride: {frame_stride} (slower analysis)")
    print(f"🖼️  Display scale: {display_scale*100}% (less zoomed)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_stride == 0:
            img_feat = encode_frame(model, preprocess, frame, device)
            sim = (100.0 * img_feat @ text_feat.T).softmax(dim=-1)
            probs = sim.squeeze(0).cpu().numpy()
            frame_probs.append(probs)
            
            # Get current top prediction for display
            best_i = np.argmax(probs)
            current_label = cls_of_idx[best_i]
            current_confidence = probs[best_i] * 100
            
            # Track scores for each category over time
            zone_a_score = sum(probs[i] for i, cls in enumerate(cls_of_idx) if "zoneA" in cls)
            zone_b_score = sum(probs[i] for i, cls in enumerate(cls_of_idx) if "zoneB" in cls)
            movement_score = sum(probs[i] for i, cls in enumerate(cls_of_idx) if "move" in cls or "transit" in cls or "hand" in cls)
            toppled_score = sum(probs[i] for i, cls in enumerate(cls_of_idx) if "topple" in cls)
            
            zone_a_scores.append(zone_a_score)
            zone_b_scores.append(zone_b_score)
            movement_scores.append(movement_score)
            toppled_scores.append(toppled_score)
            
            # 🎯 DYNAMIC UPDATES - Show progress as it processes
            timestamp = frame_idx / fps
            
            # Print when confident or scene changes (like nature analyzer)
            if (prev_top_label != current_label and current_confidence > 40.0) or (frames_used % 3 == 0):  # ← More frequent updates
                print(f"⏱ {timestamp:6.1f}s | {current_label:.<35} | {current_confidence:5.1f}%")
                prev_top_label = current_label
            
            # Real-time display with overlay - LESS ZOOMED
            if can_show:
                overlay = frame.copy()
                # Show current prediction
                cv2.putText(overlay, f"{current_label} ({current_confidence:.1f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay, f"Time: {timestamp:.1f}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay, f"Frame: {frames_used}/{max_frames}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # RESIZE to be less zoomed
                height, width = overlay.shape[:2]
                new_width = int(width * display_scale)
                new_height = int(height * display_scale)
                resized_frame = cv2.resize(overlay, (new_width, new_height))
                
                try:
                    cv2.imshow("Mug Action Analysis - Live", resized_frame)
                    # SLOWER DISPLAY: Add a small delay to make it easier to watch
                    wait_time = 1 if frame_stride <= 6 else 1  # Adjust as needed
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break
                except Exception:
                    # Headless cases
                    pass
            
            frames_used += 1
            if frames_used >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if can_show:
        cv2.destroyAllWindows()

    if not frame_probs:
        return None, {"frames_used": 0, "fps": fps}

    # Analyze temporal patterns
    temporal_analysis = analyze_temporal_patterns(
        zone_a_scores, zone_b_scores, movement_scores, toppled_scores, fps, frame_stride
    )
    
    agg = agg_probs_over_video(frame_probs)
    return agg, {"frames_used": frames_used, "fps": fps, "temporal": temporal_analysis}



def analyze_temporal_patterns(zone_a, zone_b, movement, toppled, fps, stride):
    """Analyze how scores change over time to detect movement patterns"""
    
    # Find dominant zones at start and end
    start_window = 5  # First 5 samples
    end_window = 5    # Last 5 samples
    
    avg_zone_a_start = np.mean(zone_a[:start_window])
    avg_zone_b_start = np.mean(zone_b[:start_window])
    avg_zone_a_end = np.mean(zone_a[-end_window:])
    avg_zone_b_end = np.mean(zone_b[-end_window:])
    
    # Detect movement peaks
    movement_peak = np.max(movement)
    avg_movement = np.mean(movement)
    
    # Analyze transitions
    start_zone = "A" if avg_zone_a_start > avg_zone_b_start else "B"
    end_zone = "A" if avg_zone_a_end > avg_zone_b_end else "B"
    
    movement_detected = movement_peak > 0.3  # Threshold for movement confidence
    zone_changed = start_zone != end_zone
    
    return {
        "start_zone": start_zone,
        "end_zone": end_zone,
        "zone_changed": zone_changed,
        "movement_detected": movement_detected,
        "movement_confidence": movement_peak,
        "avg_zone_a_start": avg_zone_a_start,
        "avg_zone_b_start": avg_zone_b_start,
        "avg_zone_a_end": avg_zone_a_end,
        "avg_zone_b_end": avg_zone_b_end,
    }

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Zero-shot mug action recognition with OpenCLIP")
    ap.add_argument("--videos_dir", type=str, default="videos", help="Folder with MOV/MP4 files")
    ap.add_argument("--video", type=str, help="Path to a single video file")
    ap.add_argument("--labels", type=str, default=None, help="Optional labels.json path")
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--stride", type=int, default=24, help="process every Nth frame")
    ap.add_argument("--max_frames", type=int, default=400, help="max frames sampled per video")
    ap.add_argument("--extensions", type=str, nargs="+", default=[".mov", ".mp4", ".MOV", ".MP4"])
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"🔧 Using device: {device}")

    print(f"📦 Loading model: {args.model} ({args.pretrained})")
    model, preprocess, tokenizer = load_model(args.model, args.pretrained, device)

    print("📝 Loading labels…")
    labels, prompts, cls_of_idx = load_labels(args.labels)

    print("🔤 Encoding prompts…")
    # Get the proper tokenizer for the model
    clip_tokenizer = open_clip.get_tokenizer(args.model)
    text_feat = encode_texts(model, clip_tokenizer, prompts, device)  # ← CORRECT

    # ---- Single-file mode takes precedence ----
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ File not found: {args.video}")
            return
        files = [os.path.basename(args.video)]
        args.videos_dir = os.path.dirname(args.video) or "."
    else:
        # Folder mode
        if not os.path.isdir(args.videos_dir):
            print(f"❌ No such folder: {args.videos_dir}")
            return
        files = [f for f in os.listdir(args.videos_dir) if any(f.endswith(ext) for ext in args.extensions)]
        files.sort()
        if not files:
            print(f"❌ No videos found in {args.videos_dir}")
            return

    print(f"🎥 Found {len(files)} video(s).")
    for fname in files:
        vpath = os.path.join(args.videos_dir, fname)
        
        print(f"\n🎬 Starting analysis: {fname}")
        print("=" * 70)
        
        agg, info = analyze_video_temporal(
            vpath, model, preprocess, text_feat, cls_of_idx, prompts,
            device=device, frame_stride=args.stride, max_frames=args.max_frames
        )

        if agg is None:
            print("❌ No frames processed.")
            continue

        # Final summary (like nature analyzer)
        top = summarize_class_probs(agg, cls_of_idx)
        print(f"\n📊 FINAL RESULTS - Frames used: {info['frames_used']} | FPS: {info['fps']:.1f}")
        print("-" * 70)
        
        # Show temporal analysis
        temporal = info.get('temporal', {})
        if temporal:
            print("🕒 TEMPORAL ANALYSIS:")
            print(f"   Start zone: {temporal.get('start_zone', 'N/A')}")
            print(f"   End zone: {temporal.get('end_zone', 'N/A')}")
            print(f"   Zone changed: {temporal.get('zone_changed', False)}")
            print(f"   Movement detected: {temporal.get('movement_detected', False)}")
            print("-" * 70)
        
        for cls, p in top[:6]:
            print(f"{cls:28s} -> {p*100:5.1f}%")
        print("-" * 70)

        # Enhanced action prediction
        best_cls, best_p = top[0]
        temporal_info = info.get('temporal', {})
        action = determine_action(best_cls, best_p, temporal_info)
        print(f"✅ FINAL PREDICTION: {best_cls}  (conf {best_p*100:.1f}%)")
        print(f"🤖 ACTION: {action}")
        print("=" * 70)

        

def determine_action(best_class, confidence, temporal_info):
    """Determine action based on both frame classification and temporal patterns"""
    
    if temporal_info.get('zone_changed', False) and temporal_info.get('movement_detected', False):
        start_zone = temporal_info['start_zone']
        end_zone = temporal_info['end_zone']
        return f"MUG_MOVED_{start_zone}_TO_{end_zone}"
    
    if "topple" in best_class.lower() and confidence > 40:
        return "PICK_UP_AND_STAND_UP"
    
    if "zoneA" in best_class and confidence > 40:
        return "MUG_STABLE_IN_ZONE_A"
        
    if "zoneB" in best_class and confidence > 40:
        return "MUG_STABLE_IN_ZONE_B"
    
    return "NO_OPERATION_NEEDED"

if __name__ == "__main__":
    main()
