import os, io, argparse
import torch
import open_clip
import cv2
from PIL import Image
import numpy as np

def pick_device(req: str = "auto") -> str:
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        return "cuda"  # may error if kernels aren’t supported
    # auto
    if torch.cuda.is_available():
        try:
            _ = torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            return "cuda"
        except Exception:
            pass
    return "cpu"

def download_youtube_video(url, output_path="nature_timelapse.mp4"):
    """
    Download a YouTube video (prefer yt-dlp, fallback to pytube).
    Returns the output path or None on failure.
    """
    # 1) Try yt-dlp
    try:
        import yt_dlp
        print("📥 Downloading via yt-dlp ...")
        # Prefer ≤720p MP4; fall back gracefully
        ydl_opts = {
            "outtmpl": output_path,  # exact filename
            "format": (
                "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
                "best[height<=720][ext=mp4]/best"
            ),
            "merge_output_format": "mp4",  # needs ffmpeg for bestvideo+bestaudio
            "quiet": False,
            "noprogress": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ Download complete: {output_path}")
        return output_path
    except ImportError:
        print("ℹ️ yt-dlp not installed; skipping to pytube.")
    except Exception as e:
        print(f"⚠️ yt-dlp failed: {e}. Trying pytube...")

    # 2) Fallback: pytube progressive MP4
    try:
        from pytube import YouTube
        print("📥 Downloading via pytube ...")
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension='mp4', progressive=True, res="720p").first()
        if not stream:
            stream = yt.streams.filter(file_extension='mp4', progressive=True).order_by('resolution').desc().first()
        stream.download(filename=output_path)
        print(f"✅ Download complete: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Download failed with pytube: {e}")
        return None


def analyze_nature_timelapse(video_path, frame_interval=30, display_video=True, device="auto"):
    # ---- Model ----
    print("🚀 Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    device = pick_device(device)
    print(f"🔧 Using device: {device}")
    model = model.to(device).eval()

    # ---- Prompts (you can edit/expand) ----
    # ---- Prompts (with confusers to test system intelligence) ----
    prompts = [
        # Astronomy (real)
        "milky way galaxy night sky", "starry night", "astrophotography",
        "night sky with stars", "constellations in sky",
        
        # Weather (real)
        "storm clouds supercell", "lightning storm", "dramatic weather clouds",
        "thunderstorm clouds", "tornado formation",
        
        # Atmospherics (real)
        "rainbow in sky", "foggy mountain landscape", "misty forest",
        "clouds moving fast", "sunset with colorful clouds",
        
        # Landscapes (real)
        "desert rock formations", "monument valley landscape",
        "california coastline", "utah desert scenery",
        "mountain range landscape", "canyon landscape",
        
        # General (real)
        "time lapse of nature", "landscape photography",
        "outdoor scenery", "nature documentary footage",
        
        # 🎯 CONFUSER PROMPTS (should get low scores)
        "indoor office space", "person using computer", "city traffic jam",
        "shopping mall interior", "kitchen cooking scene", 
        "bedroom with furniture", "subway station", "airport terminal",
        "shopping cart in store", "office desk with monitor",
        "library bookshelves", "restaurant dining room",
        "car interior dashboard", "classroom with students",
        "bathroom with toilet", "supermarket aisle",
        "video game screenshot", "mobile phone screen",
        "text document", "spreadsheet on computer",
        "programming code editor", "math equations on whiteboard"
    ]
    text_tokens = tokenizer(prompts).to(device)

    # ---- Video ----
    print("🎥 Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
    duration = total_frames / fps if total_frames else 0
    print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} sec")
    print(f"🔍 Processing every {frame_interval} frames...")
    print("-" * 60)

    frame_idx = 0
    prev_label = None
    results = []

    # Small helper to safely show frames
    can_show = display_video and (os.environ.get("DISPLAY") or os.name == "nt")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # BGR -> RGB -> preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            image_tensor = preprocess(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                if device == "cuda":
                    with torch.amp.autocast("cuda"):
                        img_feat = model.encode_image(image_tensor)
                        txt_feat = model.encode_text(text_tokens)
                else:
                    img_feat = model.encode_image(image_tensor)
                    txt_feat = model.encode_text(text_tokens)

                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].detach().cpu().numpy()

            best_i = int(np.argmax(probs))
            best_p = float(probs[best_i] * 100.0)
            label = prompts[best_i]
            t = frame_idx / fps

            # Print when confident or scene changes
            if (prev_label != label and best_p > 60.0) or (frame_idx % (frame_interval * 5) == 0):
                print(f"⏱ {t:6.1f}s | {label:.<35} | {best_p:5.1f}%")
                prev_label = label

            results.append({
                "frame": frame_idx,
                "time_s": t,
                "label": label,
                "confidence": best_p,
                "probs": probs.tolist(),
            })

            if can_show:
                overlay = frame.copy()
                cv2.putText(overlay, f"{label} ({best_p:.1f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay, f"Time: {t:.1f}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                try:
                    cv2.imshow("OpenCLIP Nature Timelapse Analysis", overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    # Headless cases
                    pass

        frame_idx += 1

    cap.release()
    if can_show:
        cv2.destroyAllWindows()

    summarize(results, prompts)
    return results

def summarize(results, prompts):
    print("\n" + "="*60)
    print("📊 OPENCLIP ANALYSIS SUMMARY")
    print("="*60)

    if not results:
        print("No samples analyzed.")
        return

    # counts & averages
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in results:
        buckets[r["label"]].append(r["confidence"])

    print(f"\n🎬 SCENE BREAKDOWN ({len(results)} samples):")
    print("-" * 40)
    for label, confs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        avg = sum(confs) / len(confs)
        print(f"  {label:.<35} {len(confs):3d} frames | avg: {avg:5.1f}%")

    highs = sorted([r for r in results if r["confidence"] > 70.0],
                   key=lambda r: -r["confidence"])[:10]
    if highs:
        print(f"\n⭐ HIGH CONFIDENCE DETECTIONS (>70%):")
        print("-" * 40)
        for r in highs:
            print(f"  {r['time_s']:6.1f}s | {r['label']:.<35} {r['confidence']:5.1f}%")

    # scene transitions (first 8 unique)
    order = []
    last = None
    for r in results:
        if r["label"] != last:
            if r["label"] not in order:
                order.append(r["label"])
            last = r["label"]
    if order:
        print(f"\n🔄 SCENE TRANSITIONS:")
        print("-" * 40)
        print("  " + " → ".join(order[:8]))


    # Add confuser analysis
    confuser_keywords = ["office", "computer", "mall", "kitchen", "desk", "store", "subway", "airport", "library", "restaurant", "car", "classroom", "bathroom", "supermarket", "video game", "phone", "document", "spreadsheet", "programming", "math"]
    
    confuser_results = []
    normal_results = []
    
    for r in results:
        is_confuser = any(keyword in r["label"].lower() for keyword in confuser_keywords)
        if is_confuser:
            confuser_results.append(r)
        else:
            normal_results.append(r)
    
    print(f"\n🎭 CONFUSER ANALYSIS:")
    print("-" * 40)
    print(f"  Normal prompts: {len(normal_results)} samples")
    print(f"  Confuser prompts: {len(confuser_results)} samples")
    
    if confuser_results:
        avg_confuser_conf = sum(r["confidence"] for r in confuser_results) / len(confuser_results)
        avg_normal_conf = sum(r["confidence"] for r in normal_results) / len(normal_results)
        print(f"  Avg confidence - Normal: {avg_normal_conf:.1f}%")
        print(f"  Avg confidence - Confusers: {avg_confuser_conf:.1f}%")
        
        # Show worst confuser matches (highest false positives)
        high_confusers = [r for r in confuser_results if r["confidence"] > 30]
        if high_confusers:
            print(f"\n  ⚠️  FALSE POSITIVES (confusers with >30% confidence):")
            for r in sorted(high_confusers, key=lambda x: -x["confidence"])[:5]:
                print(f"    {r['time_s']:6.1f}s | {r['label']:.<35} {r['confidence']:5.1f}%")
    


def main():
    ap = argparse.ArgumentParser(description="Analyze nature timelapses with OpenCLIP")
    ap.add_argument('--video', type=str, help='Path to existing video file')
    ap.add_argument('--youtube', type=str, help='YouTube URL to download')
    ap.add_argument('--interval', type=int, default=30, help='Process every Nth frame (bigger=faster)')
    ap.add_argument('--no-display', action='store_true', help='Disable video display overlay')
    ap.add_argument('--device', type=str, choices=["auto","cpu","cuda"], default="auto",
                    help='Device to use (default: auto)')
    args = ap.parse_args()

    video_path = args.video
    if args.youtube and not video_path:
        video_path = download_youtube_video(args.youtube)
        if not video_path:
            return

    if not video_path:
        print("❌ Provide --video or --youtube")
        return
    if not os.path.exists(video_path):
        print(f"❌ Not found: {video_path}")
        return

    analyze_nature_timelapse(video_path, frame_interval=args.interval,
                             display_video=not args.no_display, device=args.device)

if __name__ == "__main__":
    main()
