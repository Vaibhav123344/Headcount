import redis
import json
import msgpack # ── Upgrade 4: Binary Serialization
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
    # --- FIX 1: Add local_id to arguments ---
    def __init__(self, global_id, local_id, embedding, last_seen, last_x, last_y, cam_id, velocity_x=0.0, velocity_y=0.0):
        self.global_id  = global_id
        self.feature_bank = [embedding] 
        self.last_seen   = last_seen
        self.last_x      = last_x
        self.last_y      = last_y
        self.velocity_x  = velocity_x
        self.velocity_y  = velocity_y
        self.cam_history = {cam_id}
        
        # --- FIX 1 (cont): Save actual local_id ---
        self.active_local_ids = {cam_id: int(local_id)} 
        self.last_cam_update = {cam_id: last_seen}

class GlobalMatcher:
    def __init__(self):
        cfg = load_config()
        self.r = redis.Redis(host=cfg['redis']['host'], port=cfg['redis']['port'], db=0, socket_timeout=None)
        self.in_queue = "warehouse:queue:matcher"
        
        m_cfg = cfg['matcher']
        self.time_buffer_sec    = m_cfg['time_buffer_ms'] / 1000.0  # ── Upgrade 1: Jitter buffer size
        self.max_speed_mps      = m_cfg['max_speed_mps']
        self.lost_short_sec     = m_cfg['lost_short_term_sec']
        self.lost_mid_sec       = m_cfg['lost_mid_term_sec']
        self.reid_sim_threshold = m_cfg.get('reid_sim_threshold', 0.60)
        
        self._gallery = {}
        self._lost_gallery = {}
        self._cam_local_to_global = {}
        self._next_id = 1
        self._jitter_buffer = []  # Holds raw messages waiting to be aligned

    @staticmethod
    def _bank_similarity(feature_bank, query_emb):
        """Matches query against the best view of the person."""
        return max(float(np.dot(emb, query_emb)) for emb in feature_bank)

    def _time_aligned_distance(self, data, entry):
        time_diff = max(0, data['timestamp'] - entry.last_seen)
        projected_x = entry.last_x + entry.velocity_x * time_diff
        projected_y = entry.last_y + entry.velocity_y * time_diff
        return math.hypot(data['world_x'] - projected_x, data['world_y'] - projected_y)

    # ── Upgrade 1: Jitter Buffer / Time Windowing ──
    def _gather_and_align_windows(self):
        """Pulls from Redis, sorts by time, and yields perfectly aligned temporal windows."""
        # Drain all currently available messages quickly
        while True:
            res = self.r.lpop(self.in_queue)
            if not res: break
            self._jitter_buffer.append(msgpack.unpackb(res, strict_map_key=False))
        
        if not self._jitter_buffer:
            # Wait for at least one message if buffer is empty
            res = self.r.blpop(self.in_queue, timeout=1)
            if res: self._jitter_buffer.append(msgpack.unpackb(res[1], strict_map_key=False))

        # Sort buffer chronologically
        self._jitter_buffer.sort(key=lambda x: x['timestamp'])

        current_time = time.time()
        ready_windows = []

        # Only process items that are older than our Jitter Buffer window size
        while self._jitter_buffer and (current_time - self._jitter_buffer[0]['timestamp'] >= self.time_buffer_sec):
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
            emb = np.array(data['embedding'], dtype=np.float32)
            data['features_norm'] = emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb
            
            key = (data['cam_id'], int(data['local_track_id']))
            if key in self._cam_local_to_global and self._cam_local_to_global[key] in self._gallery:
                self._update_entry(self._cam_local_to_global[key], data)
            else:
                unmatched_items.append(data)
                
        if unmatched_items:
            still_unmatched = self._stage1_hungarian(unmatched_items)
            still_unmatched = self._stage_cascade_lost_gallery(still_unmatched) # ── Upgrade 5
            for data in still_unmatched: self._create_new_id(data)
            self._cross_camera_dedup()

    def _stage1_hungarian(self, items):
        if not self._gallery: return items
        
        gallery_gids = list(self._gallery.keys())
        cost_matrix = np.full((len(items), len(gallery_gids)), 1000.0, dtype=np.float32)
        
        for i, data in enumerate(items):
            cam_id = data['cam_id']
            local_id = int(data['local_track_id'])
            
            for j, gid in enumerate(gallery_gids):
                entry = self._gallery[gid]
                
                # --- FIX 3: Immune to Queue Lag ---
                if cam_id in entry.active_local_ids and entry.active_local_ids[cam_id] != local_id:
                    # Use absolute difference between stream timestamps!
                    time_diff = abs(data['timestamp'] - entry.last_cam_update.get(cam_id, 0))
                    if time_diff < 1.5:
                        continue # They are both in the frame at the same time. Cost stays 1000.0.

                spatial_dist = self._time_aligned_distance(data, entry)
                time_diff_safe = max(0.5, data['timestamp'] - entry.last_seen)
                
                if (spatial_dist / time_diff_safe) > self.max_speed_mps or spatial_dist > 3.0: 
                    continue
                
                sim = self._bank_similarity(entry.feature_bank, data['features_norm'])
                if sim >= 0.60:
                    cost_matrix[i, j] = (1.0 - sim) * 0.6 + (spatial_dist / 3.0) * 0.4
                
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        assigned_rows = set()
        
        for row, col in zip(row_inds, col_inds):
            if cost_matrix[row, col] < 0.35:
                data, gid = items[row], gallery_gids[col]
                self._cam_local_to_global[(data['cam_id'], int(data['local_track_id']))] = gid
                self._update_entry(gid, data)
                self.r.hset("global_id_map", f"{data['cam_id']}:{data['local_track_id']}", gid)
                assigned_rows.add(row)

        return [items[i] for i in range(len(items)) if i not in assigned_rows]

    # ── Upgrade 5: Spatiotemporal Cascade Match for Lost IDs ──
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
                    if sim > 0.80 and sim > best_sim:  # Stricter visual requirement
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
                if entry_a.cam_history == entry_b.cam_history: continue # Don't merge same-cam
                
                # --- FIX 4: Dedup Queue Lag Fix ---
                conflict = False
                shared_cams = set(entry_a.active_local_ids.keys()).intersection(set(entry_b.active_local_ids.keys()))
                for cam in shared_cams:
                    if entry_a.active_local_ids[cam] != entry_b.active_local_ids[cam]:
                        # Are they alive at the same time? Use absolute stream timestamps!
                        time_diff = abs(entry_a.last_cam_update[cam] - entry_b.last_cam_update[cam])
                        if time_diff < 2.0: # 2.0 seconds window for extra safety
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
            data['local_track_id'], # <--- FIX 2: Pass local track id here
            data['features_norm'], 
            data['timestamp'], 
            data['world_x'], 
            data['world_y'], 
            data['cam_id'], 
            data.get('velocity_x', 0), 
            data.get('velocity_y', 0)
        )
        self._cam_local_to_global[(data['cam_id'], int(data['local_track_id']))] = gid
        self.r.hset("global_id_map", f"{data['cam_id']}:{data['local_track_id']}", gid)
        log.info(f"New global_id={gid}")

    def _update_entry(self, gid, data):
        e = self._gallery[gid]
        sim_to_best = self._bank_similarity(e.feature_bank, data['features_norm'])
        
        if sim_to_best < 0.90:  
            e.feature_bank.append(data['features_norm'])
            if len(e.feature_bank) > 5: e.feature_bank.pop(0)

        e.last_seen, e.last_x, e.last_y = data['timestamp'], data['world_x'], data['world_y']
        e.velocity_x, e.velocity_y = data.get('velocity_x', 0), data.get('velocity_y', 0)
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