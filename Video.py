# Install necessary packages for both parts
print("INSTALLING BASE DEPENDENCIES...")

# --- Step 1: Uninstall ---
print("Uninstalling potentially conflicting libraries...")
!pip uninstall -y torch torchvision torchaudio ultralytics fastai openai-whisper whisper onnx onnxruntime onnxruntime-gpu transformers sentencepiece tokenizers optimum optimum[onnxruntime] optimum[onnxruntime-gpu]

# --- Step 2: Upgrade pip ---
print("Upgrading pip...")
!pip install -qU pip

# --- Step 3: Install PyTorch ---
COLAB_CUDA_VERSION_TAG = "cu121" # For Colab CUDA 12.5 driver, cu121 is compatible
print(f"Installing PyTorch, torchvision, torchaudio for {COLAB_CUDA_VERSION_TAG}...")
!pip install -qU torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{COLAB_CUDA_VERSION_TAG}

# --- Step 4: Verify PyTorch ---
print("Verifying PyTorch CUDA setup...")
import torch # Import now as it's installed
print(f"  Torch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Torch CUDA (compiled for): {torch.version.cuda}")
    if torch.cuda.device_count() > 0:
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("  No CUDA devices found by PyTorch, though CUDA might be available.")
else:
    print("  WARNING: CUDA NOT available to PyTorch. Ensure Colab runtime is GPU and restarted after changes.")


# --- Step 5: Install other core dependencies ---
print("Installing ONNX, ONNX Runtime, Transformers, Ultralytics etc....")
!pip install -qU onnx onnxruntime-gpu # Ensure onnxruntime-gpu for CUDA
!pip install -qU transformers sentencepiece tokenizers
!pip install -qU openai-whisper
!pip install -qU yt-dlp opencv-python psutil py-cpuinfo ultralytics

# --- Step 6: Install system dependencies (ffmpeg) ---
print("Installing ffmpeg...")
!sudo apt-get update -qq # -qq for quieter update
!sudo apt-get install -y ffmpeg --quiet

print("Base dependencies installed.")

# Main Script Imports (after installations and PyTorch check)
import os
import time
import glob
import copy
import sys
import json

# PyTorch related imports again, in case kernel was restarted or for clarity
import torch
import torchvision
from torchvision import transforms, models as torchvision_models # Alias to avoid conflict
import torchaudio

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic, QuantType

import whisper # openai-whisper
from ultralytics import YOLO # Capital YOLO for Ultralytics v8+

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, WhisperProcessor

import cv2
import numpy as np
from PIL import Image # Use PIL.Image to avoid conflict if cv2.Image exists
import yt_dlp
import psutil


# Helper functions
def benchmark_model(model_name, inference_func, sample_input, num_runs=20, warmup_runs=3): # Reduced runs
    print(f"\n--- Benchmarking: {model_name} ---"); latencies = []; first_output = None
    for _ in range(warmup_runs):
        try: _ = inference_func(sample_input)
        except Exception: pass
    for i in range(num_runs):
        try:
            start_time = time.perf_counter(); output = inference_func(sample_input); end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            if i == 0: first_output = output
        except Exception as e_t: print(f"Timed run err {model_name}: {e_t}"); return np.nan, None
    if not latencies: return np.nan, None
    avg_lat=np.mean(latencies); std_lat=np.std(latencies); print(f"Avg Lat: {avg_lat:.2f} ± {std_lat:.2f} ms"); return avg_lat, first_output

def get_model_size_mb(path_or_obj):
    if isinstance(path_or_obj, str):
        if os.path.isdir(path_or_obj): # For Optimum models saved as directories
            total_size = 0;
            for dirpath, _, filenames in os.walk(path_or_obj):
                for f in filenames: fp = os.path.join(dirpath, f); total_size += os.path.getsize(fp)
            return total_size / (1024*1024)
        elif os.path.exists(path_or_obj): return os.path.getsize(path_or_obj)/(1024*1024)
    elif isinstance(path_or_obj, torch.nn.Module): # For PyTorch models
        # Estimate size based on state_dict for consistency, as saving whole model can be large
        fp="tmp_torch_model_statedict.pth"; torch.save(path_or_obj.state_dict(), fp); s=os.path.getsize(fp)/(1024*1024); os.remove(fp); return s
    return 0

device_type = "cuda" if torch.cuda.is_available() else "cpu"; torch_device = torch.device(device_type)
print(f"Script will use device: {torch_device}")
OPTIMIZED_MODELS_DIR = "optimized_models"; os.makedirs(OPTIMIZED_MODELS_DIR, exist_ok=True)
sample_yolo_in_np = np.random.rand(1, 3, 640, 640).astype(np.float32)
dummy_audio_path_p1 = "dummy_audio_p1.wav"
if not os.path.exists(dummy_audio_path_p1): torchaudio.save(dummy_audio_path_p1, torch.randn(1,16000*2).cpu(), 16000)
try: dummy_pil_img_git_p1 = Image.fromarray(np.uint8(np.random.rand(384,384,3)*255))
except: dummy_pil_img_git_p1 = Image.new("RGB", (384,384))

print(f"\n{'='*20} PART 1: Model Selection, Optimization, and Benchmarking {'='*20}")

# --- 1.A Scene Description (GIT) ---
print(f"\n{'--'*10} Scene Description (GIT) {'--'*10}")
git_hf_name="microsoft/git-base"
git_pt_proc=AutoProcessor.from_pretrained(git_hf_name)
git_pt_tok=AutoTokenizer.from_pretrained(git_hf_name)
git_pt_model=AutoModelForCausalLM.from_pretrained(git_hf_name).eval().to(torch_device)
git_sample_in_pt = git_pt_proc(images=dummy_pil_img_git_p1, return_tensors="pt").pixel_values.to(torch_device)
def infer_git_pt(px_val):
    with torch.no_grad(): gen_ids=git_pt_model.generate(pixel_values=px_val,max_length=30, num_beams=3); return git_pt_tok.batch_decode(gen_ids,skip_special_tokens=True)
lat_git_pt,out_git_pt=benchmark_model("GIT (PT)",infer_git_pt,git_sample_in_pt)
size_git_pt=get_model_size_mb(git_pt_model); print(f"GIT PT (StateDict) Size: {size_git_pt:.2f} MB")
print(f"GIT PT Out: {out_git_pt[0] if out_git_pt and out_git_pt[0] else 'N/A'}")
print("NOTE: Using PyTorch GIT model for Part 2 inference.")
optimized_git_info={"type":"pytorch","model":git_pt_model,"proc":git_pt_proc,"tok":git_pt_tok, "hf_name": git_hf_name}

# --- 1.B Object Detection (YOLOv5n) ---
print(f"\n{'--'*10} Object Detection (YOLOv5n) {'--'*10}")
yolo_name_base="yolov5n"; yolo_ult_model=YOLO(f"{yolo_name_base}.pt")
yolo_pt_path = yolo_ult_model.ckpt_path if hasattr(yolo_ult_model,'ckpt_path') and yolo_ult_model.ckpt_path else f"{yolo_name_base}.pt"
if not os.path.exists(yolo_pt_path) and hasattr(yolo_ult_model, 'model') and hasattr(yolo_ult_model.model, 'yaml_file') and yolo_ult_model.model.yaml_file:
    yolo_pt_path = str(yolo_ult_model.model.yaml_file).replace(".yaml",".pt")
if not os.path.exists(yolo_pt_path) and os.path.exists(f"{yolo_name_base}u.pt"): yolo_pt_path=f"{yolo_name_base}u.pt" # Fallback for u variant
if not os.path.exists(yolo_pt_path) and os.path.exists(yolo_name_base + ".pt"): yolo_pt_path=yolo_name_base + ".pt" # Ensure default exists after YOLO() call

print(f"YOLO .pt path (for size): {yolo_pt_path}")
def infer_yolo_pt(img): return yolo_ult_model(img,verbose=False)
lat_yolo_pt,out_yolo_pt=benchmark_model(f"{yolo_name_base} (PT/Ultra)",infer_yolo_pt,np.uint8(np.random.rand(640,640,3)*255))
size_yolo_pt=get_model_size_mb(yolo_pt_path) if os.path.exists(yolo_pt_path) else 0.0; print(f"YOLO PT Size: {size_yolo_pt:.2f} MB")

onnx_yolo_fp32_p=os.path.join(OPTIMIZED_MODELS_DIR,f"{yolo_name_base}_fp32.onnx")
onnx_yolo_int8_p=os.path.join(OPTIMIZED_MODELS_DIR,f"{yolo_name_base}_int8_dynamic.onnx")
# This path is for Part 1's ONNX attempt; Part 2 will use torch.hub for YOLO inference
yolo_onnx_path_from_part1_ref = None # Renamed to avoid confusion

try:
    exp_path_obj=yolo_ult_model.export(format="onnx",opset=12,dynamic=True,half=False,simplify=True)
    act_exp_path=str(exp_path_obj); print(f"Ultra exported ONNX to: {act_exp_path}")
    if os.path.exists(act_exp_path):
        # Ensure target directory exists before renaming
        os.makedirs(os.path.dirname(onnx_yolo_fp32_p), exist_ok=True)
        os.rename(act_exp_path,onnx_yolo_fp32_p); print(f"Moved YOLO ONNX to: {onnx_yolo_fp32_p}")
        size_yolo_onnx32=get_model_size_mb(onnx_yolo_fp32_p); print(f"YOLO ONNX FP32 Size: {size_yolo_onnx32:.2f} MB")
        ort_sess_yolo32=ort.InferenceSession(onnx_yolo_fp32_p,providers=['CUDAExecutionProvider' if device_type=='cuda' else 'CPUExecutionProvider'])
        yolo32_in_name=ort_sess_yolo32.get_inputs()[0].name
        def infer_yolo_onnx32(img_np): return ort_sess_yolo32.run(None,{yolo32_in_name:img_np})
        benchmark_model(f"{yolo_name_base} (ONNX FP32)",infer_yolo_onnx32,sample_yolo_in_np)
        yolo_onnx_path_from_part1_ref = onnx_yolo_fp32_p

        print(f"Attempting dynamic quantization for YOLO ONNX: {onnx_yolo_fp32_p} -> {onnx_yolo_int8_p}")
        ort_quantize_dynamic(model_input=onnx_yolo_fp32_p,model_output=onnx_yolo_int8_p,weight_type=QuantType.QInt8)
        size_yolo_onnx8=get_model_size_mb(onnx_yolo_int8_p); print(f"YOLO ONNX INT8 Size: {size_yolo_onnx8:.2f} MB")
        try:
            ort_sess_yolo8=ort.InferenceSession(onnx_yolo_int8_p,providers=['CPUExecutionProvider'])
            yolo8_in_name=ort_sess_yolo8.get_inputs()[0].name
            def infer_yolo_onnx8(img_np): return ort_sess_yolo8.run(None,{yolo8_in_name:img_np})
            benchmark_model(f"{yolo_name_base} (ONNX INT8 CPU)",infer_yolo_onnx8,sample_yolo_in_np)
            yolo_onnx_path_from_part1_ref = onnx_yolo_int8_p
            print(f"INT8 YOLO ONNX model seems usable on CPU: {yolo_onnx_path_from_part1_ref}")
        except Exception as e_yolo_int8_load:
            print(f"ERR loading/running YOLO INT8 ONNX: {e_yolo_int8_load}. FP32 ONNX for reference: {yolo_onnx_path_from_part1_ref}")
    else: print(f"ERR: Exported ONNX {act_exp_path} not found.")
except Exception as e: print(f"ERR processing YOLO: {e}"); yolo_onnx_path_from_part1_ref = onnx_yolo_fp32_p if os.path.exists(onnx_yolo_fp32_p) else None
print(f"Reference ONNX YOLO from Part 1: {yolo_onnx_path_from_part1_ref}")


# --- 1.C Speech (Whisper-tiny) ---
print(f"\n{'--'*10} Speech (Whisper-tiny) {'--'*10}")
wh_name="tiny"; wh_pt_model_obj=whisper.load_model(wh_name,device=torch_device)
wh_processor = WhisperProcessor.from_pretrained(f"openai/whisper-{wh_name}")
def infer_wh_pt(apath): return wh_pt_model_obj.transcribe(apath,fp16=(torch_device.type=='cuda'))
lat_wh_pt,out_wh_pt=benchmark_model(f"Whisper-{wh_name} (PT)",infer_wh_pt,dummy_audio_path_p1)
size_wh_pt_approx = get_model_size_mb(wh_pt_model_obj.encoder) + get_model_size_mb(wh_pt_model_obj.decoder)
print(f"Whisper-{wh_name} PT (Enc+Dec StateDicts) Approx. Size: {size_wh_pt_approx:.2f} MB")
print("NOTE: Using PyTorch Whisper for Part 2 inference.")
optimized_whisper_info = {"type":"pytorch", "model_object":wh_pt_model_obj, "name":wh_name, "processor":wh_processor, "hf_name": f"openai/whisper-{wh_name}"}

SELECTED_SCENE_MODEL_INFO = optimized_git_info
# For Part 2, we will use torch.hub for YOLO for simplicity and robustness
SELECTED_OBJECT_LOAD_METHOD = "torch.hub"
SELECTED_OBJECT_MODEL_NAME_HUB = 'yolov5n' # The specific model name for torch.hub
SELECTED_WHISPER_INFO = optimized_whisper_info
print(f"\nFinal Selections for Part 2:\n  Scene: GIT ({SELECTED_SCENE_MODEL_INFO['type']})\n  Object: YOLOv5n ({SELECTED_OBJECT_LOAD_METHOD})\n  Speech: Whisper-{SELECTED_WHISPER_INFO['name']} ({SELECTED_WHISPER_INFO['type']})")

# =====================================================================================
print(f"\n{'='*20} PART 2: Inference Pipeline with Optimized Models {'='*20}")
# =====================================================================================
git_inf_model=SELECTED_SCENE_MODEL_INFO["model"]; git_inf_proc=SELECTED_SCENE_MODEL_INFO["proc"]; git_inf_tok=SELECTED_SCENE_MODEL_INFO["tok"]
print("GIT model for scene description loaded (PyTorch).")

# Load YOLOv5n using torch.hub for inference in Part 2
yolo_torch_hub_model = None
yolo_active_for_inference = False
print(f"Loading YOLO ({SELECTED_OBJECT_MODEL_NAME_HUB}) via torch.hub for Part 2 inference...")
try:
    # Ensure ultralytics/yolov5 cache is clear for torch.hub to avoid issues if local clone was bad
    # torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub_alt")) # Optional: force different cache
    yolo_torch_hub_model = torch.hub.load('ultralytics/yolov5', SELECTED_OBJECT_MODEL_NAME_HUB, pretrained=True, trust_repo=True, force_reload=False).to(torch_device).eval()
    print(f"YOLO ({SELECTED_OBJECT_MODEL_NAME_HUB}) via torch.hub loaded successfully.")
    yolo_active_for_inference = True
except Exception as e:
    print(f"Error loading YOLO via torch.hub: {e}. Object detection might not work.")
    # Attempt to load without trust_repo as a fallback, though less secure
    try:
        print("Attempting YOLO torch.hub load without trust_repo...")
        yolo_torch_hub_model = torch.hub.load('ultralytics/yolov5', SELECTED_OBJECT_MODEL_NAME_HUB, pretrained=True, force_reload=False).to(torch_device).eval()
        print(f"Fallback YOLO ({SELECTED_OBJECT_MODEL_NAME_HUB}) via torch.hub loaded successfully.")
        yolo_active_for_inference = True
    except Exception as e2:
        print(f"Fallback YOLO torch.hub load also failed: {e2}")


wh_inf_model_p2=SELECTED_WHISPER_INFO["model_object"]
wh_processor_p2 = SELECTED_WHISPER_INFO.get("processor")
print(f"Using Whisper-{SELECTED_WHISPER_INFO['name']} ({SELECTED_WHISPER_INFO['type']}) for Part 2 speech.")

vid_url="https://www.youtube.com/watch?v=pO63rsqlRBw&ab_channel=Seeker"; dl_vid_path="inf_vid.mp4"
aud_wh_path="inf_aud_wh.wav"; aud_merge_path="inf_aud_merge.aac"
vid_only_out_path="cap_vid_only_tmp.mp4"; final_vid_out_path="cap_vid_final_audio.mp4"
print(f"\nDownloading video for inference: {vid_url}...");
ydl_opts = {'format': 'best[ext=mp4][height<=480]', 'outtmpl': dl_vid_path, 'quiet': True, 'noplaylist': True, 'overwrites': True}
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([vid_url])
except Exception as e_dl: print(f"Error downloading video: {e_dl}")
if os.path.exists(dl_vid_path):
    print(f"Extracting audio for Whisper to {aud_wh_path}...")
    os.system(f"ffmpeg -y -i {dl_vid_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {aud_wh_path} -loglevel error")
    print(f"Extracting/Converting audio for merging to {aud_merge_path}...")
    os.system(f"ffmpeg -y -i {dl_vid_path} -vn -c:a aac -b:a 128k {aud_merge_path} -loglevel error")
else:
    print(f"Video file {dl_vid_path} not found. Cannot proceed."); aud_wh_path=None; aud_merge_path=None; sys.exit()

speech_segs=[]
if wh_inf_model_p2 and aud_wh_path and os.path.exists(aud_wh_path) and os.path.getsize(aud_wh_path)>0:
    print("Transcribing audio...");
    try:
        res=wh_inf_model_p2.transcribe(aud_wh_path,fp16=(wh_inf_model_p2.device.type=='cuda'))
        speech_segs=res['segments']; print(f"Transcription done, {len(speech_segs)} segs.")
    except Exception as e: print(f"Whisper err: {e}")
else: print("Skipping Whisper.")

if not os.path.exists(dl_vid_path): print(f"ERR: Vid {dl_vid_path} not found."); sys.exit()
cap=cv2.VideoCapture(dl_vid_path)
if not cap.isOpened(): print(f"ERR: Could not open vid {dl_vid_path}"); sys.exit()
fps=cap.get(cv2.CAP_PROP_FPS); fr_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fr_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
act_fps=fps if fps>0 else 30.0
out_vid_only=cv2.VideoWriter(vid_only_out_path,cv2.VideoWriter_fourcc(*'mp4v'),act_fps,(fr_w,fr_h))
print(f"Processing video: {fr_w}x{fr_h} @ {act_fps:.2f} FPS. Temp out: {vid_only_out_path}")
fr_idx=0; proc_every_n=max(1,int(act_fps/2))
last_scene_txt="Scene: Detecting..."; last_obj_txt="Objects: Detecting..."
font=cv2.FONT_HERSHEY_SIMPLEX; font_sc_ov=0.55; font_th_ov=1; lt=cv2.LINE_AA

def draw_wrap_txt(img,txt,org,font,fsc,col,fth,ltype,max_w,bg_col=None,y_inc_f=1.3):
    lines=[];cline="";total_h=0
    if not txt: return img,0
    words=txt.split();
    if not words: return img,0
    for word in words:
        tline=f"{cline} {word}".strip();(tw,th),bl=cv2.getTextSize(tline,font,fsc,fth)
        if tw>max_w and cline: lines.append(cline); cline=word
        else: cline=tline
    if cline: lines.append(cline)
    x,y_s=org;cur_y=y_s
    for i,ltxt in enumerate(lines):
        (lw,lh),bl=cv2.getTextSize(ltxt,font,fsc,fth); l_y_draw=cur_y+lh
        if bg_col: cv2.rectangle(img,(x-2,cur_y-2),(x+lw+2,cur_y+lh+bl+2),bg_col,-1)
        cv2.putText(img,ltxt,(x,l_y_draw),font,fsc,col,fth,ltype); cur_y+=int(lh*y_inc_f)+2
        if i==len(lines)-1: total_h=(l_y_draw-y_s)
    return img,total_h

while cap.isOpened():
    succ,fr_bgr=cap.read()
    if not succ: break
    cur_ts=fr_idx/act_fps; disp_fr=fr_bgr.copy(); speech_txt=""
    if speech_segs:
        for seg in speech_segs:
            if seg['start']<=cur_ts<=seg['end']: speech_txt=seg['text'].strip(); break

    if fr_idx%proc_every_n==0:
        if git_inf_model and git_inf_proc and git_inf_tok:
            try:
                pil_img=Image.fromarray(cv2.cvtColor(fr_bgr,cv2.COLOR_BGR2RGB))
                inputs=git_inf_proc(images=pil_img,return_tensors="pt").pixel_values.to(git_inf_model.device)
                with torch.no_grad(): gen_ids=git_inf_model.generate(pixel_values=inputs,max_length=20,num_beams=3,early_stopping=True,no_repeat_ngram_size=2)
                scene_gen_txt=git_inf_tok.batch_decode(gen_ids,skip_special_tokens=True)[0]
                last_scene_txt=f"Scene: {scene_gen_txt.capitalize() if scene_gen_txt else 'No descr.'}"
            except Exception as e: last_scene_txt=f"Scene: Err ({type(e).__name__})"
        else: last_scene_txt="Scene: N/A"

        obj_stat_fr="Objects: Detecting..."
        if yolo_active_for_inference and yolo_torch_hub_model:
            try:
                yolo_res = yolo_torch_hub_model(fr_bgr)
                det_df = yolo_res.pandas().xyxy[0]
                obj_list = list(det_df['name'])

                # Draw boxes from torch.hub model directly
                for _, row in det_df.iterrows():
                    x1, y1, x2, y2, conf, name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
                    if conf > 0.3: # Confidence threshold for drawing
                        cv2.rectangle(disp_fr, (x1, y1), (x2, y2), (0, 255, 0), 1) # Green box
                        cv2.putText(disp_fr, f"{name}", (x1, y1 - 5), font, 0.4, (0, 255, 0), 1, lt)

                obj_stat_fr = f"Objects: {', '.join(list(set(obj_list[:3])))}" if obj_list else "Objects: None"
            except Exception as e_yolo_hub:
                obj_stat_fr = f"Objects: Err ({type(e_yolo_hub).__name__})"
        else:
            obj_stat_fr = "Objects: Model N/A (Hub)"
        last_obj_txt=obj_stat_fr

    txt_y_s=30; txt_x_s=30; l_space=5; bg_col_ov=(0,0,0)
    disp_fr,h_sc=draw_wrap_txt(disp_fr,last_scene_txt,(txt_x_s,txt_y_s),font,font_sc_ov,(255,255,0),font_th_ov,lt,fr_w-(2*txt_x_s),bg_col_ov)
    cur_y_txt=txt_y_s+(h_sc if h_sc > 0 else cv2.getTextSize("T",font,font_sc_ov,font_th_ov)[0][1]) + l_space
    disp_fr,h_obj=draw_wrap_txt(disp_fr,last_obj_txt,(txt_x_s,cur_y_txt),font,font_sc_ov,(0,255,255),font_th_ov,lt,fr_w-(2*txt_x_s),bg_col_ov)
    if speech_txt:
        sp_y_s_draw=fr_h-20; max_sw=fr_w-60; slines=[]; cline=""
        for word in speech_txt.split():
            tline=f"{cline} {word}".strip();(stw,sth),_=cv2.getTextSize(tline,font,font_sc_ov,font_th_ov)
            if stw>max_sw and cline: slines.append(cline); cline=word
            else: cline=tline
        if cline: slines.append(cline)
        for i,ltxt in enumerate(reversed(slines)):
            (lw,lh_txt),_=cv2.getTextSize(ltxt,font,font_sc_ov,font_th_ov); l_ypos=sp_y_s_draw-i*(lh_txt+8)
            if l_ypos-lh_txt<0: break
            cv2.rectangle(disp_fr,(txt_x_s-2,l_ypos-lh_txt-2),(txt_x_s+lw+2,l_ypos+5),bg_col_ov,-1)
            cv2.putText(disp_fr,ltxt,(txt_x_s,l_ypos),font,font_sc_ov,(255,255,255),font_th_ov,lt)

    out_vid_only.write(disp_fr); fr_idx+=1
    if fr_idx%(int(act_fps)*10)==0: print(f"Proc {fr_idx} fr ({cur_ts:.1f}s)... Sc: '{last_scene_txt[:30]}...', Obj: '{last_obj_txt[:30]}...'")
cap.release(); out_vid_only.release(); cv2.destroyAllWindows()
print(f"✅ Vid (only) proc done. Temp: {vid_only_out_path}")

if os.path.exists(vid_only_out_path) and \
   aud_merge_path and os.path.exists(aud_merge_path) and os.path.getsize(aud_merge_path)>0:
    print(f"Merging vid '{vid_only_out_path}' with aud '{aud_merge_path}'...")
    merge_cmd=f"ffmpeg -y -i {vid_only_out_path} -i {aud_merge_path} -c:v copy -c:a copy -shortest {final_vid_out_path} -loglevel error"
    ret_code=os.system(merge_cmd)
    if ret_code==0 and os.path.exists(final_vid_out_path) and os.path.getsize(final_vid_out_path)>0:
        print(f"✅ Final vid with aud: {final_vid_out_path}")
        if os.path.exists(vid_only_out_path): os.remove(vid_only_out_path)
        if os.path.exists(aud_wh_path): os.remove(aud_wh_path)
        if os.path.exists(aud_merge_path): os.remove(aud_merge_path)
    else: print(f"ERR: Merge failed (ret {ret_code}). Vid w/o aud: {vid_only_out_path}")
else: print(f"Skipping audio merge. Vid only: {vid_only_out_path} (Audio for merge missing/empty).")
print(f"\n{'='*20} SCRIPT FINISHED {'='*20}")