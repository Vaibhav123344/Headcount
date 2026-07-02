import redis
import json
import msgpack
import numpy as np
import time
import math
import logging
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("global_matcher")

def load_config():
    with open("config.json") as f:
        return json.load(f)

class GalleryEntry:
    def __init__(self, global_id, local_id, embedding, last_seen, last_x, last_y, cam_id, velocity_x=0.0, velocity_y=0.0):
        self.global_id  = global_id
        self.feature_bank = [embedding] 
        self.last_seen   = last_seen
        self.last_x      = last_x
        self.last_y      = last_y
        self.velocity_x  = velocity_x
        self.velocity_y  = velocity_y
        self.cam_history = {cam_id}
        
        # Save actual local_id to track stream presence 
        self.active_local_ids = {cam_id: int(local_id)} 
        self.last_cam_update = {cam_id: last_seen}

class GlobalMatcher:
    def __init__(self):
        cfg = load_config()
        self.r = redis.Redis(host=cfg['redis']['host'], port=cfg['redis']['port'], db=0, socket_timeout=None)
        self.in_queue = "warehouse:queue:matcher"
        
        m_cfg = cfg['matcher']
        self.time_buffer_sec    = m_cfg['time_buffer_ms'] / 1000.0  
        self.max_speed_mps      = m_cfg['max_speed_mps']
        self.lost_short_sec     = m_cfg['lost_short_term_sec']
        self.lost_mid_sec       = m_cfg['lost_mid_term_sec']
        self.reid_sim_threshold = m_cfg.get('reid_sim_threshold', 0.60)
        
        self._gallery = {}
        self._lost_gallery = {}
        self._cam_local_to_global = {}
        self._next_id = 1
        self._jitter_buffer = []  

    @staticmethod
    def _bank_similarity(feature_bank, query_emb):
        """Matches query against the best view of the person."""
        return max(float(np.dot(emb, query_emb)) for emb in feature_bank)

    def _time_aligned_distance(self, data, entry):
        time_diff = max(0, data['timestamp'] - entry.last_seen)
        projected_x = entry.last_x + entry.velocity_x * time_diff
        projected_y = entry.last_y + entry.velocity_y * time_diff
        return math.hypot(data['world_x'] - projected_x, data['world_y'] - projected_y)

    @staticmethod
    def _normalize_keys(data):
        """Convert all msgpack byte-keys to string-keys."""
        return {
            (k.decode('utf-8') if isinstance(k, bytes) else k): v
            for k, v in data.items()
        }

    def _gather_and_align_windows(self):
        """Pulls from Redis, sorts by time, and yields perfectly aligned temporal windows."""
        # Drain all currently available messages quickly
        while True:
            res = self.r.lpop(self.in_queue)
            if not res: break
            self._jitter_buffer.append(self._normalize_keys(msgpack.unpackb(res, strict_map_key=False)))
        
        if not self._jitter_buffer:
            # Wait for at least one message if buffer is empty
            res = self.r.blpop(self.in_queue, timeout=1)
            if res: self._jitter_buffer.append(self._normalize_keys(msgpack.unpackb(res[1], strict_map_key=False)))

        if not self._jitter_buffer:
            return []

        # Sort buffer chronologically
        self._jitter_buffer.sort(key=lambda x: x['timestamp'])

        # FIX 1: Use the highest stream timestamp as current time. 
        # Makes the buffer immune to system clock drift and queue lag.
        latest_stream_time = self._jitter_buffer[-1]['timestamp']
        ready_windows = []

        # Only process items that are older than our Jitter Buffer window size
        while self._jitter_buffer and (latest_stream_time - self._jitter_buffer[0]['timestamp'] >= self.time_buffer_sec):
            window_start_time = self._jitter_buffer[0]['timestamp']
            
            # Find all items that fall within this exact time window
            window_items = []
            while self._jitter_buffer and self._jitter_buffer[0]['timestamp'] <= window_start_time + self.time_buffer_sec:
                window_items.append(self._jitter_buffer.pop(0))
            
            ready_windows.append(window_items)

        return ready_windows

    def process_window(self, batch):
        unmatched_items = []
        for data in batch:
            # Defensive: skip items that arrived without an embedding key entirely
            if 'embedding' not in data:
                continue

            emb = np.array(data['embedding'], dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            has_real_embedding = emb_norm > 0.01  # Zero embeddings from ReID worker have norm ~0

            if has_real_embedding:
                data['features_norm'] = emb / emb_norm
            else:
                data['features_norm'] = emb  # Keep as zero vector
            
            key = (data['cam_id'], int(data['local_track_id']))
            if key in self._cam_local_to_global and self._cam_local_to_global[key] in self._gallery:
                # ALREADY TRACKED: Always update position/velocity.
                # Only update the feature bank if we have a real visual embedding.
                self._update_entry(self._cam_local_to_global[key], data, update_features=has_real_embedding)
            else:
                # UNMATCHED: Only attempt matching/creation if we have a real visual embedding.
                # Without a real embedding, we can't visually match, and creating a new ID
                # with a zero-vector would poison the gallery. Wait for the next ReID cycle.
                if has_real_embedding:
                    unmatched_items.append(data)
                
        if unmatched_items:
            still_unmatched = self._stage1_hungarian(unmatched_items)
            still_unmatched = self._stage_cascade_lost_gallery(still_unmatched) 
            for data in still_unmatched: self._create_new_id(data)
            self._cross_camera_dedup()

    def _calculate_sota_probabilistic_cost(self, incoming_data, gallery_entry):
        """
        Implements SOTA Probabilistic Spatiotemporal Fusion using Time-Decaying Gaussian Uncertainty.
        Returns a cost (0.0 to 1.0) for the Hungarian algorithm. Lower is better.
        """
        dt = incoming_data['timestamp'] - gallery_entry.last_seen
        dt = max(0.01, dt) 

        # 1. KINEMATIC PREDICTION (Where should they be?)
        predict_dt = min(dt, 3.0)
        pred_x = gallery_entry.last_x + (gallery_entry.velocity_x * predict_dt)
        pred_y = gallery_entry.last_y + (gallery_entry.velocity_y * predict_dt)

        # 2. EUCLIDEAN DISTANCE (Error between actual and predicted)
        spatial_dist = math.hypot(incoming_data['world_x'] - pred_x, incoming_data['world_y'] - pred_y)

        # 3. DYNAMIC UNCERTAINTY (Sigma)
        sigma = 0.5 + (1.2 * dt) 

        # 4. SPATIAL PROBABILITY (Gaussian Distribution)
        p_spatial = math.exp(- (spatial_dist ** 2) / (2 * (sigma ** 2)))

        # 5. VISUAL PROBABILITY (OSNet ReID)
        raw_sim = self._bank_similarity(gallery_entry.feature_bank, incoming_data['features_norm'])
        p_visual = max(0.0, raw_sim) 

        # 6. JOINT PROBABILITY
        joint_probability = p_spatial * p_visual

        # 7. HARD GATING (Physical Limits)
        if (spatial_dist / dt) > self.max_speed_mps:
            return float('inf'), spatial_dist

        final_cost = 1.0 - joint_probability
        
        return final_cost, spatial_dist

    def _stage1_hungarian(self, items):
        if not self._gallery: return items
        
        gallery_gids = list(self._gallery.keys())
        cost_matrix = np.full((len(items), len(gallery_gids)), 1000.0, dtype=np.float32)
        
        for i, data in enumerate(items):
            cam_id = data['cam_id']
            local_id = int(data['local_track_id'])
            
            for j, gid in enumerate(gallery_gids):
                entry = self._gallery[gid]
                
                # Immune to Queue Lag
                if cam_id in entry.active_local_ids and entry.active_local_ids[cam_id] != local_id:
                    time_diff = abs(data['timestamp'] - entry.last_cam_update.get(cam_id, 0))
                    if time_diff < 1.5:
                        continue 

                cost, spatial_dist = self._calculate_sota_probabilistic_cost(data, entry)
                if cost == float('inf'):
                    continue
                
                cost_matrix[i, j] = cost
                
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        assigned_rows = set()
        
        for row, col in zip(row_inds, col_inds):
            if cost_matrix[row, col] < 0.70:
                data, gid = items[row], gallery_gids[col]
                self._cam_local_to_global[(data['cam_id'], int(data['local_track_id']))] = gid
                self._update_entry(gid, data)
                self.r.hset("global_id_map", f"{data['cam_id']}:{data['local_track_id']}", gid)
                assigned_rows.add(row)

        return [items[i] for i in range(len(items)) if i not in assigned_rows]

    def _stage_cascade_lost_gallery(self, items):
        still_unmatched = []
        now = time.time()
        
        for data in items:
            best_gid, best_sim = None, -1.0
            
            for gid, entry in self._lost_gallery.items():
                time_lost = now - entry.last_seen
                sim = self._bank_similarity(entry.feature_bank, data['features_norm'])
                
                # Cascade 1: Short term lost (Occlusion). Require spatial + visual
                if time_lost <= self.lost_short_sec:
                    spatial_dist = self._time_aligned_distance(data, entry)
                    if spatial_dist < 4.0 and sim > 0.55 and sim > best_sim:
                        best_sim, best_gid = sim, gid
                
                # Cascade 2: Mid term lost (Left room). Purely visual, strict threshold
                elif time_lost <= self.lost_mid_sec:
                    if sim > 0.80 and sim > best_sim:  
                        best_sim, best_gid = sim, gid

            if best_gid:
                entry = self._lost_gallery.pop(best_gid)
                self._gallery[best_gid] = entry
                self._cam_local_to_global[(data['cam_id'], int(data['local_track_id']))] = best_gid
                self._update_entry(best_gid, data)
                self.r.hset("global_id_map", f"{data['cam_id']}:{data['local_track_id']}", best_gid)
                log.info(f"Reclaimed lost ID {best_gid} via Cascade (sim={best_sim:.3f})")
            else:
                still_unmatched.append(data)
                
        return still_unmatched

    def _cross_camera_dedup(self):
        gids = list(self._gallery.keys())
        merged = set()
        
        for i in range(len(gids)):
            if gids[i] in merged: continue
            for j in range(i + 1, len(gids)):
                if gids[j] in merged: continue
                
                entry_a, entry_b = self._gallery[gids[i]], self._gallery[gids[j]]
                if entry_a.cam_history == entry_b.cam_history: continue 
                
                # Dedup Queue Lag Fix
                conflict = False
                shared_cams = set(entry_a.active_local_ids.keys()).intersection(set(entry_b.active_local_ids.keys()))
                for cam in shared_cams:
                    if entry_a.active_local_ids[cam] != entry_b.active_local_ids[cam]:
                        time_diff = abs(entry_a.last_cam_update[cam] - entry_b.last_cam_update[cam])
                        if time_diff < 2.0: 
                            conflict = True
                            break
                if conflict: continue

                best_sim = max(float(np.dot(ea, eb)) for ea in entry_a.feature_bank for eb in entry_b.feature_bank)
                if best_sim >= 0.55:
                    keep, drop = min(gids[i], gids[j]), max(gids[i], gids[j])
                    self._gallery[keep].feature_bank.extend(self._gallery[drop].feature_bank)
                    self._gallery[keep].feature_bank = self._gallery[keep].feature_bank[-5:]
                    self._gallery[keep].cam_history.update(self._gallery[drop].cam_history)
                    
                    for k, v in list(self._cam_local_to_global.items()):
                        if v == drop:
                            self._cam_local_to_global[k] = keep
                            self.r.hset("global_id_map", f"{k[0]}:{k[1]}", keep)
                            
                    del self._gallery[drop]
                    merged.add(drop)
                    log.info(f"Dedup: Merged {drop} into {keep}")

    def _create_new_id(self, data):
        gid, self._next_id = self._next_id, self._next_id + 1
        
        self._gallery[gid] = GalleryEntry(
            gid, 
            data['local_track_id'], 
            data['features_norm'], 
            data['timestamp'], 
            data['world_x'], 
            data['world_y'], 
            data['cam_id'], 
            0.0, # Initial velocity X
            0.0  # Initial velocity Y
        )
        self._cam_local_to_global[(data['cam_id'], int(data['local_track_id']))] = gid
        self.r.hset("global_id_map", f"{data['cam_id']}:{data['local_track_id']}", gid)
        log.info(f"New global_id={gid}")

    def _update_entry(self, gid, data, update_features=True):
        e = self._gallery[gid]

        if update_features:
            sim_to_best = self._bank_similarity(e.feature_bank, data['features_norm'])
            if sim_to_best < 0.90:  
                e.feature_bank.append(data['features_norm'])
                if len(e.feature_bank) > 5: e.feature_bank.pop(0)

        # FIX 2: Calculate real-world velocity internally
        dt = data['timestamp'] - e.last_seen
        if dt > 0.1: # Only update velocity if enough time passed to measure movement
            inst_vel_x = (data['world_x'] - e.last_x) / dt
            inst_vel_y = (data['world_y'] - e.last_y) / dt
            
            # Smooth velocity (EMA) to avoid erratic predictions from bounding box jitter
            e.velocity_x = (e.velocity_x * 0.6) + (inst_vel_x * 0.4)
            e.velocity_y = (e.velocity_y * 0.6) + (inst_vel_y * 0.4)

        e.last_seen, e.last_x, e.last_y = data['timestamp'], data['world_x'], data['world_y']
        e.cam_history.add(data['cam_id'])
        
        # --- MUTUAL EXCLUSIVITY FIX ---
        e.active_local_ids[data['cam_id']] = int(data['local_track_id'])
        e.last_cam_update[data['cam_id']] = data['timestamp']

    def run(self):
        log.info("Starting Global Matcher (Jitter Buffer + Kalman + msgpack)...")
        last_cleanup = time.time()
        
        while True:
            # Move old IDs to lost, or delete entirely
            if time.time() - last_cleanup > 2.0:
                now = time.time()
                for gid in [g for g, e in self._gallery.items() if now - e.last_seen > self.lost_short_sec]:
                    self._lost_gallery[gid] = self._gallery.pop(gid)
                    for k, v in list(self._cam_local_to_global.items()):
                        if v == gid:
                            self.r.hdel("global_id_map", f"{k[0]}:{k[1]}")
                            del self._cam_local_to_global[k]
                
                for gid in [g for g, e in self._lost_gallery.items() if now - e.last_seen > self.lost_mid_sec]:
                    del self._lost_gallery[gid]
                last_cleanup = now
                
            windows = self._gather_and_align_windows()
            for window in windows:
                self.process_window(window)
                
            # Dashboard State remains JSON for Streamlit's text parser
            now = time.time()
            snapshot = {
                "unique_people": len(self._gallery),
                "next_id": self._next_id,
                "entries": [{"global_id": e.global_id, "last_x": round(e.last_x, 2), "last_y": round(e.last_y, 2), "cameras": sorted(e.cam_history)} for e in self._gallery.values()]
            }
            self.r.set("state:gallery", json.dumps(snapshot))

if __name__ == "__main__":
    GlobalMatcher().run()