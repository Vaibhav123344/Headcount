import redis
import json
import base64
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    FeatureExtractor = None

class ReIDWorker:
    INPUT_HW = (256, 128)

    def __init__(self):
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        self.in_queue = "warehouse:queue:reid"
        self.out_queue = "warehouse:queue:matcher"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tf = transforms.Compose([
            transforms.Resize(self.INPUT_HW),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        self.model = self._build_model()
        self.model.eval().to(self.device)

    def _build_model(self):
        if FeatureExtractor is not None:
            try:
                print("Attempting to load OSNet model via torchreid...")
                extractor = FeatureExtractor(
                    model_name='osnet_x1_0',
                    model_path='osnet_x1_0_imagenet.pth',
                    device=str(self.device)
                )
                self.backend = "osnet"
                self._osnet = extractor
                print("OSNet model loaded successfully.")
                return extractor.model
            except Exception as e:
                print(f"Failed to load torchreid extractor ({e}). Falling back to torchvision ResNet50.")
        
        # ResNet50 Fallback (standard torchvision model, output dim 2048)
        print("Using ResNet50 fallback from torchvision...")
        self.backend = "resnet50"
        self._osnet = None
        from torchvision.models import resnet50, ResNet50_Weights
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        net.fc = nn.Identity() # Remove final fc layer to output raw 2048-dim feature maps
        return net

    def extract_features_batch(self, imgs):
        if not imgs:
            return []
            
        pil_imgs = []
        for img in imgs:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_imgs.append(Image.fromarray(img_rgb))
        
        if self.backend == "osnet" and self._osnet is not None:
            feats = self._osnet(pil_imgs)
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
        else:
            with torch.no_grad():
                tensor_imgs = torch.stack([self.tf(pil_img) for pil_img in pil_imgs]).to(self.device)
                feats = self.model(tensor_imgs).cpu().numpy()
                
        # L2 Normalize feature vectors
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        feats = feats / norms
        
        return feats

    def _gather_batch(self, max_batch_size, timeout):
        batch = []
        # Block until at least one item
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

    def run(self, max_batch_size=16):
        print(f"Starting Re-ID Worker (Backend: {self.backend}, Device: {self.device}, Batch Size: {max_batch_size})...")
        while True:
            messages = self._gather_batch(max_batch_size=max_batch_size, timeout=0.05)
            if not messages:
                continue
                
            batch_data = []
            valid_imgs = []
            valid_indices = []
            
            for idx, message in enumerate(messages):
                data = json.loads(message.decode('utf-8'))
                batch_data.append(data)
                
                img_data = base64.b64decode(data['image_crop'])
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is not None and img.size > 0:
                    valid_imgs.append(img)
                    valid_indices.append(idx)
                    
            if valid_imgs:
                try:
                    features_batch = self.extract_features_batch(valid_imgs)
                    for i, feat in zip(valid_indices, features_batch):
                        batch_data[i]['features'] = feat.tolist()
                except Exception as e:
                    print(f"Error extracting batch features: {e}")
                    dim = 512 if self.backend == "osnet" else 2048
                    for i in valid_indices:
                        batch_data[i]['features'] = np.zeros(dim).tolist()
                        
            # Fill empty for invalid images and push all
            dim = 512 if self.backend == "osnet" else 2048
            for data in batch_data:
                if 'features' not in data:
                    data['features'] = np.zeros(dim).tolist()
                
                del data['image_crop']
                self.r.rpush(self.out_queue, json.dumps(data))

if __name__ == "__main__":
    worker = ReIDWorker()
    worker.run()
