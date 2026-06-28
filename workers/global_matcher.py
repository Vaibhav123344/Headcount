import redis
import json
import numpy as np
import time
import math
import logging
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("global_matcher")

class GalleryEntry:
    def __init__(self, global_id, embedding, last_seen, last_x, last_y, cam_id):
        self.global_id = global_id
        self.embedding = embedding
        self.last_seen = last_seen
        self.last_x = last_x
        self.last_y = last_y
        self.cam_history = {cam_id}

class GlobalMatcher:
    def __init__(self, similarity_threshold=0.62, ema_alpha=0.1, expiry_sec=30.0, max_speed_mps=5.0):
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        self.in_queue = "warehouse:queue:matcher"
        
        self.sim_thresh = similarity_threshold
        self.ema = ema_alpha
        self.expiry = expiry_sec
        self.max_speed_mps = max_speed_mps
        
        self._gallery = {}  # global_id -> GalleryEntry
        self._cam_local_to_global = {}  # (cam_id, local_id) -> global_id
        self._next_id = 1

    def _gather_batch(self, window=0.5):
        batch = []
        # Block until we get AT LEAST ONE item
        res = self.r.blpop(self.in_queue, timeout=2)
        if not res:
            return batch
            
        batch.append(json.loads(res[1].decode('utf-8')))
        
        # Now wait for window to gather concurrent items
        end_time = time.time() + window
        while time.time() < end_time:
            res = self.r.lpop(self.in_queue)
            if res:
                batch.append(json.loads(res.decode('utf-8')))
            else:
                time.sleep(0.05)
        return batch

    def process_batch(self, batch):
        unmatched_items = []
        
        for data in batch:
            emb = np.array(data['features'], dtype=np.float32)
            # Normalize embedding
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            data['features_norm'] = emb
            
            cam_id = data['cam_id']
            local_id = int(data['local_track_id'])
            key = (cam_id, local_id)
            
            # 1. Lock: Pre-locked tracks bypass Hungarian matching
            if key in self._cam_local_to_global and self._cam_local_to_global[key] in self._gallery:
                gid = self._cam_local_to_global[key]
                self._update_entry(gid, emb, data['world_x'], data['world_y'], cam_id, data['timestamp'])
                # Publish global ID mapping immediately
                self.r.hset("global_id_map", f"{cam_id}:{local_id}", gid)
            else:
                unmatched_items.append(data)
                
        if not unmatched_items:
            return
            
        # 2. Hungarian Matching for new/unlocked tracks
        if not self._gallery:
            # Gallery is empty, everything is new
            for data in unmatched_items:
                self._create_new_id(data)
            return
            
        gallery_gids = list(self._gallery.keys())
        num_items = len(unmatched_items)
        num_gallery = len(gallery_gids)
        
        # Initialize Cost Matrix with very high cost (1000.0)
        cost_matrix = np.full((num_items, num_gallery), 1000.0, dtype=np.float32)
        
        for i, data in enumerate(unmatched_items):
            for j, gid in enumerate(gallery_gids):
                entry = self._gallery[gid]
                
                # Spatial gating
                time_diff = max(0.01, data['timestamp'] - entry.last_seen)
                spatial_dist = math.hypot(data['world_x'] - entry.last_x, data['world_y'] - entry.last_y)
                speed = spatial_dist / time_diff
                
                if speed > self.max_speed_mps:
                    continue # Cost remains 1000.0 (blocked)
                    
                # Cosine similarity (1.0 - sim = cost)
                sim = float(np.dot(entry.embedding, data['features_norm']))
                cost_matrix[i, j] = 1.0 - sim
                
        # Solve Hungarian Algorithm
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        
        assigned_items = set()
        
        for row, col in zip(row_inds, col_inds):
            cost = cost_matrix[row, col]
            data = unmatched_items[row]
            
            # If the best match meets our similarity threshold
            if cost < (1.0 - self.sim_thresh):
                gid = gallery_gids[col]
                cam_id = data['cam_id']
                local_id = int(data['local_track_id'])
                
                log.info(f"Batched Match: global_id={gid} (cam={cam_id} local={local_id} similarity={1.0 - cost:.3f})")
                
                self._cam_local_to_global[(cam_id, local_id)] = gid
                self._update_entry(gid, data['features_norm'], data['world_x'], data['world_y'], cam_id, data['timestamp'])
                self.r.hset("global_id_map", f"{cam_id}:{local_id}", gid)
                
                assigned_items.add(row)
                
        # 3. Create new IDs for unassigned items
        for i, data in enumerate(unmatched_items):
            if i not in assigned_items:
                self._create_new_id(data)
                
    def _create_new_id(self, data):
        gid = self._next_id
        self._next_id += 1
        cam_id = data['cam_id']
        local_id = int(data['local_track_id'])
        
        self._gallery[gid] = GalleryEntry(gid, data['features_norm'], data['timestamp'], data['world_x'], data['world_y'], cam_id)
        self._cam_local_to_global[(cam_id, local_id)] = gid
        self.r.hset("global_id_map", f"{cam_id}:{local_id}", gid)
        
        log.info(f"New global_id={gid} (cam={cam_id} local={local_id})")

    def _update_entry(self, gid, emb, world_x, world_y, cam_id, ts):
        e = self._gallery[gid]
        
        # Smoothen appearance embedding dynamically using EMA (prevents sudden lighting noise)
        new_emb = (1.0 - self.ema) * e.embedding + self.ema * emb
        nn = np.linalg.norm(new_emb)
        if nn > 0:
            new_emb = new_emb / nn
            
        e.embedding = new_emb
        e.last_seen = ts
        e.last_x = world_x
        e.last_y = world_y
        e.cam_history.add(cam_id)

    def cleanup(self):
        now = time.time()
        # Expire tracks that haven't been seen in expiry_sec
        stale = [gid for gid, e in self._gallery.items() if now - e.last_seen > self.expiry]
        for gid in stale:
            log.info(f"Expiring global_id={gid} due to inactivity.")
            del self._gallery[gid]
            for k, v in list(self._cam_local_to_global.items()):
                if v == gid:
                    del self._cam_local_to_global[k]

    def run(self):
        log.info("Starting Batched Global Matcher (Hungarian + Spatial gating)...")
        last_cleanup = time.time()
        
        while True:
            # Periodically cleanup expired tracks
            if time.time() - last_cleanup > 5.0:
                self.cleanup()
                last_cleanup = time.time()
                
            batch = self._gather_batch(window=0.5)
            if batch:
                self.process_batch(batch)
                
                # Publish gallery snapshot for Streamlit dashboard
                now = time.time()
                snapshot = {
                    "unique_people": len(self._gallery),
                    "next_id": self._next_id,
                    "entries": [{
                        "global_id": e.global_id,
                        "last_x": round(e.last_x, 2),
                        "last_y": round(e.last_y, 2),
                        "age_sec": round(now - e.last_seen, 1),
                        "cameras": sorted(e.cam_history),
                    } for e in self._gallery.values()]
                }
                self.r.set("state:gallery", json.dumps(snapshot))

if __name__ == "__main__":
    matcher = GlobalMatcher()
    matcher.run()
