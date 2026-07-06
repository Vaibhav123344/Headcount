import json
import redis
import msgpack
import cv2
import numpy as np
import time
import torch
import timm
from PIL import Image

# Let cuDNN pick the fastest kernels for our fixed input size.
torch.backends.cudnn.benchmark = True

class ModernReIDWorker:
    # ================= MODEL CONFIGURATION — SWITCH HERE =================
    # Exactly ONE of the next two lines must be active.
    #
    #   siglip = current default (generic ViT, robust to lighting/color).
    #   osnet  = person-ReID model (MSMT17), more discriminative + faster.
    #            After switching to osnet you MUST re-run tools/redis_reid_tuner.py
    #            and update reid_sim_threshold in config.json (scale differs!).
    #
    # To use OSNet:  COMMENT the siglip line, UNCOMMENT the osnet line.
    BACKBONE_TYPE = "siglip"
    #BACKBONE_TYPE = "osnet"
    # ====================================================================

    MODELS = {
        "siglip": "vit_base_patch16_siglip_224",
        "swin": "swin_base_patch4_window7_224"
    }
    OSNET_WEIGHTS = "osnet_x1_0_msmt17.pth"

    def __init__(self):
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        
        # Intercepts from SCTWorker, pushes to GlobalMatcher
        self.in_queue = "warehouse:queue:reid"
        self.out_queue = "warehouse:queue:reid_result"
        
        # Hardware acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == "cuda"

        # Drop crops that arrived too late to be worth embedding (backlog guard).
        try:
            self.stale_reid_sec = json.load(open("config.json")).get("matcher", {}).get("stale_reid_sec", 3.0)
        except Exception:
            self.stale_reid_sec = 3.0

        # Build model and dynamic transforms
        self.model, self.tf, self.embed_dim = self._build_model()

    def _build_model(self):
        if self.BACKBONE_TYPE == "osnet":
            return self._build_osnet()

        model_name = self.MODELS.get(self.BACKBONE_TYPE, self.MODELS["siglip"])
        print(f"Loading {self.BACKBONE_TYPE.upper()} backbone: {model_name}...")
        
        try:
            # num_classes=0 strips the classification head, returning raw feature embeddings
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
            model.eval().to(self.device)
            
            # Dynamically pull the exact normalization, mean/std, and image size required by this specific ViT
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            
            # Pass a dummy tensor to determine exact output embedding dimension (e.g., 768 or 1024)
            dummy_input = torch.randn(1, 3, data_config['input_size'][1], data_config['input_size'][2]).to(self.device)
            with torch.no_grad():
                dummy_out = model(dummy_input)
                embed_dim = dummy_out.shape[1]
                
            print(f"Model loaded successfully. Embedding Dimension: {embed_dim}")
            return model, transforms, embed_dim
            
        except Exception as e:
            print(f"Failed to load ViT model ({e}). Ensure 'timm' is installed.")
            raise

    def _build_osnet(self):
        """Person-ReID backbone: OSNet x1.0 (MSMT17). 512-d features, 256x128 input."""
        import torchreid
        from torchvision import transforms as T
        print(f"Loading OSNET backbone: osnet_x1_0 ({self.OSNET_WEIGHTS})...")
        model = torchreid.models.build_model("osnet_x1_0", num_classes=1000, pretrained=False)
        torchreid.utils.load_pretrained_weights(model, self.OSNET_WEIGHTS)
        model.eval().to(self.device)  # eval() mode: forward returns 512-d features, not logits
        tf = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        embed_dim = 512
        print(f"OSNet loaded successfully. Embedding Dimension: {embed_dim}")
        return model, tf, embed_dim

    def extract_features_batch(self, imgs):
        if not imgs: 
            return []
            
        # Convert OpenCV BGR to PIL RGB
        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs]
        
        with torch.inference_mode():
            # Apply dynamic transforms and stack into a single batch tensor
            tensor_imgs = torch.stack([self.tf(pil_img) for pil_img in pil_imgs]).to(self.device)

            # Forward pass through the Transformer (fp16 autocast on GPU for ~2x speed;
            # weights stay fp32, output cast back to fp32 before normalize).
            if self.use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    feats = self.model(tensor_imgs)
                feats = feats.float().cpu().numpy()
            else:
                feats = self.model(tensor_imgs).cpu().numpy()

        # L2 Normalize feature vectors (Critical for cosine similarity matching)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1  
        feats = feats / norms
        
        return feats

    @staticmethod
    def _normalize_keys(data):
        """Convert all msgpack byte-keys to string-keys to prevent key mismatch bugs."""
        return {
            (k.decode('utf-8') if isinstance(k, bytes) else k): v
            for k, v in data.items()
        }

    def _gather_batch(self, max_batch_size, timeout):
        batch = []
        res = self.r.blpop(self.in_queue, timeout=1)
        if not res: 
            return batch
            
        batch.append(res[1])
        end_time = time.time() + timeout
        
        while len(batch) < max_batch_size and time.time() < end_time:
            res = self.r.lpop(self.in_queue)
            if res:
                batch.append(res)
            else:
                time.sleep(0.01)
        return batch

    def run(self, max_batch_size=32):
        # A batch size of 32-64 easily clears ViT-Base on an RTX 5090 without stalling
        print(f"Starting ReID Middleware (Device: {self.device}, Batch Size: {max_batch_size})...")
        
        # Signal to the pipeline launcher that the model is loaded and we are ready
        self.r.set("reid_worker:ready", "1")
        print("ReID Worker signaled READY to pipeline.")
        
        # Create the zero-embedding once, reuse forever
        zero_embedding = np.zeros(self.embed_dim).tolist()
        
        while True:
            messages = self._gather_batch(max_batch_size=max_batch_size, timeout=0.05)
            if not messages:
                continue
                
            batch_data = []
            valid_imgs = []
            valid_indices = []
            
            now = time.time()
            for idx, message in enumerate(messages):
                raw_data = msgpack.unpackb(message, strict_map_key=False)
                # Normalize ALL keys to strings immediately to kill byte-key bugs
                data = self._normalize_keys(raw_data)

                # Backlog guard: skip crops too old to matter (don't waste ViT compute).
                if self.stale_reid_sec > 0 and 'timestamp' in data and (now - data['timestamp']) > self.stale_reid_sec:
                    continue

                if "image_bytes" in data:
                    img_data = data.pop("image_bytes") # Remove massive byte array from payload to save Redis memory
                    np_arr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if img is not None and img.size > 0:
                        valid_imgs.append(img)
                        valid_indices.append(idx)
                        
                batch_data.append(data)
                    
            if valid_imgs:
                try:
                    features_batch = self.extract_features_batch(valid_imgs)
                    for i, feat in zip(valid_indices, features_batch):
                        # Global Matcher expects the key strictly as "embedding"
                        batch_data[i]['embedding'] = feat.tolist()
                except Exception as e:
                    print(f"Error extracting batch features: {e}")
                    for i in valid_indices:
                        batch_data[i]['embedding'] = zero_embedding
                        
            # BULLETPROOF: Ensure EVERY payload has an embedding before pushing
            for data in batch_data:
                if "embedding" not in data:
                    data['embedding'] = zero_embedding
                
                self.r.rpush(self.out_queue, msgpack.packb(data))

if __name__ == "__main__":
    worker = ModernReIDWorker()
    worker.run(max_batch_size=32)