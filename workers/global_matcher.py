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
    def __init__(self, global_id, local_id, timestamp, last_x, last_y, cam_id, zone="Unknown"):
        self.global_id  = global_id
        self.feature_bank = [] # Added later by reid_result
        self.last_seen   = timestamp
        self.last_x      = last_x
        self.last_y      = last_y
        self.velocity_x  = 0.0
        self.velocity_y  = 0.0
        self.cam_history = {cam_id}
        self.zone = zone
        
        self.active_local_ids = {cam_id: int(local_id)} 
        self.last_cam_update = {cam_id: timestamp}

class GlobalMatcher:
    def __init__(self):
        self.cfg = load_config()
        self.r = redis.Redis(host=self.cfg['redis']['host'], port=self.cfg['redis']['port'], db=0, socket_timeout=None)
        
        self.in_queue = "warehouse:queue:matcher"
        self.reid_queue = "warehouse:queue:reid_result"
        
        m_cfg = self.cfg['matcher']
        self.time_buffer_sec    = m_cfg['time_buffer_ms'] / 1000.0  
        self.max_speed_mps      = m_cfg['max_speed_mps']
        self.lost_short_sec     = m_cfg['lost_short_term_sec']
        self.lost_mid_sec       = m_cfg['lost_mid_term_sec']
        self.reid_sim_threshold = m_cfg.get('reid_sim_threshold', 0.60)
        self.spatial_match_radius_m = m_cfg.get('spatial_match_radius_m', 2.0)
        self.dedup_spatial_radius_m = m_cfg.get('dedup_spatial_radius_m', 2.0)
        
        l_cfg = self.cfg.get('layout', {})
        self.pixels_per_meter = l_cfg.get('pixels_per_meter', 50.0)
        self.zones = l_cfg.get('zones', [])
        
        self._gallery = {}
        self._lost_gallery = {}
        self._cam_local_to_global = {}
        self._next_id = 1
        self._jitter_buffer = []  

    def _get_zone(self, x, y):
        for z in self.zones:
            if z['x1'] <= x <= z['x2'] and z['y1'] <= y <= z['y2']:
                return z['name']
        return "Unknown"

    @staticmethod
    def _normalize_keys(data):
        return {
            (k.decode('utf-8') if isinstance(k, bytes) else k): v
            for k, v in data.items()
        }

    def _poll_reid_results(self):
        """Poll warehouse:queue:reid_result to update feature banks."""
        while True:
            res = self.r.lpop(self.reid_queue)
            if not res: break
            data = self._normalize_keys(msgpack.unpackb(res, strict_map_key=False))
            key = (data['cam_id'], int(data['local_track_id']))
            if key in self._cam_local_to_global:
                gid = self._cam_local_to_global[key]
                if gid in self._gallery and 'embedding' in data:
                    emb = np.array(data['embedding'], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0.01:
                        emb = emb / norm
                        self._gallery[gid].feature_bank.append(emb)
                        if len(self._gallery[gid].feature_bank) > 5:
                            self._gallery[gid].feature_bank.pop(0)

    def _gather_and_align_windows(self):
        while True:
            res = self.r.lpop(self.in_queue)
            if not res: break
            self._jitter_buffer.append(self._normalize_keys(msgpack.unpackb(res, strict_map_key=False)))
        
        if not self._jitter_buffer:
            res = self.r.blpop(self.in_queue, timeout=1)
            if res: self._jitter_buffer.append(self._normalize_keys(msgpack.unpackb(res[1], strict_map_key=False)))

        if not self._jitter_buffer:
            return []

        self._jitter_buffer.sort(key=lambda x: x['timestamp'])
        latest_stream_time = self._jitter_buffer[-1]['timestamp']
        ready_windows = []

        while self._jitter_buffer and (latest_stream_time - self._jitter_buffer[0]['timestamp'] >= self.time_buffer_sec):
            window_start_time = self._jitter_buffer[0]['timestamp']
            window_items = []
            while self._jitter_buffer and self._jitter_buffer[0]['timestamp'] <= window_start_time + self.time_buffer_sec:
                window_items.append(self._jitter_buffer.pop(0))
            ready_windows.append(window_items)

        return ready_windows


    def _distance_m(self, x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2) / self.pixels_per_meter

    def _bank_similarity(self, bank1, bank2):
        if not bank1 or not bank2:
            return 0.0
        return max(float(np.dot(e1, e2)) for e1 in bank1 for e2 in bank2)

    def process_window(self, batch):
        active_in_batch = {}
        for data in batch:
            cam_id = data['cam_id']
            if cam_id not in active_in_batch:
                active_in_batch[cam_id] = set()
            active_in_batch[cam_id].add(int(data['local_track_id']))

        unmatched_items = []
        for data in batch:
            key = (data['cam_id'], int(data['local_track_id']))
            if key in self._cam_local_to_global and self._cam_local_to_global[key] in self._gallery:
                self._update_entry(self._cam_local_to_global[key], data)
            else:
                unmatched_items.append(data)
                
        if unmatched_items:
            still_unmatched = self._stage1_spatial_match(unmatched_items, active_in_batch)
            still_unmatched = self._stage2_recover_lost(still_unmatched)
            for data in still_unmatched: 
                self._create_new_id(data)
            
            # Diagnostic: log all cross-camera distances before dedup
            if self._next_id <= 20:  # Only on early windows
                cam_entries = {}
                for gid, entry in self._gallery.items():
                    cam = max(entry.last_cam_update, key=entry.last_cam_update.get)
                    if cam not in cam_entries:
                        cam_entries[cam] = []
                    cam_entries[cam].append((gid, entry))
                cams = list(cam_entries.keys())
                if len(cams) >= 2:
                    for ga_gid, ga_e in cam_entries[cams[0]]:
                        for gb_gid, gb_e in cam_entries[cams[1]]:
                            d = self._distance_m(ga_e.last_x, ga_e.last_y, gb_e.last_x, gb_e.last_y)
                            log.info(f"  DIAG: {cams[0]}:gid{ga_gid}({ga_e.last_x:.0f},{ga_e.last_y:.0f}) <-> {cams[1]}:gid{gb_gid}({gb_e.last_x:.0f},{gb_e.last_y:.0f}) = {d:.1f}m")
            
            self._cross_camera_dedup(active_in_batch)

    def _stage1_spatial_match(self, items, active_in_batch):
        if not self._gallery: return items
        
        gallery_gids = list(self._gallery.keys())
        cost_matrix = np.full((len(items), len(gallery_gids)), 1000.0, dtype=np.float32)
        
        for i, data in enumerate(items):
            cam_id = data['cam_id']
            world_x, world_y = data['world_x'], data['world_y']
            local_id = int(data['local_track_id'])
            
            for j, gid in enumerate(gallery_gids):
                entry = self._gallery[gid]
                
                # Prevent matching to an old track that is still simultaneously active
                if cam_id in entry.active_local_ids:
                    old_local_id = entry.active_local_ids[cam_id]
                    if old_local_id != local_id and old_local_id in active_in_batch.get(cam_id, set()):
                        continue 
                        
                dist = self._distance_m(world_x, world_y, entry.last_x, entry.last_y)
                
                # Calculate speed required to move there
                dt = max(0.01, data['timestamp'] - entry.last_seen)
                req_speed = dist / dt
                
                # Reject impossible physics unless they are fundamentally close enough for calibration error
                if req_speed > self.max_speed_mps and dist > 2.0:
                    continue
                    
                # Strict distance limit (4.0m) to prevent random jumping
                if dist <= 4.0:
                    cost_matrix[i, j] = dist

        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        assigned_rows = set()
        
        for row, col in zip(row_inds, col_inds):
            if cost_matrix[row, col] <= 4.0:
                data, gid = items[row], gallery_gids[col]
                self._cam_local_to_global[(data['cam_id'], int(data['local_track_id']))] = gid
                self._update_entry(gid, data)
                self.r.hset("global_id_map", f"{data['cam_id']}:{data['local_track_id']}", gid)
                assigned_rows.add(row)
                
        return [items[i] for i in range(len(items)) if i not in assigned_rows]

    def _stage2_recover_lost(self, items):
        """Recover tracks from lost gallery using spatial proximity + optional ReID."""
        if not self._lost_gallery or not items:
            return items

        still_unmatched = []
        for data in items:
            world_x, world_y = data['world_x'], data['world_y']
            best_gid, best_score = None, float('inf')

            for gid, entry in self._lost_gallery.items():
                dist = self._distance_m(world_x, world_y, entry.last_x, entry.last_y)
                
                # Must be within reasonable distance (3m) to recover
                if dist > 3.0:
                    continue

                # If we have ReID features, use them to boost confidence
                score = dist
                if entry.feature_bank and 'features_norm' in data:
                    sim = self._bank_similarity(entry.feature_bank, data['features_norm'])
                    if sim > 0.4:
                        score = dist * (1.0 - sim)  # Lower score = better match

                if score < best_score:
                    best_score = score
                    best_gid = gid

            if best_gid is not None and best_score < 3.0:
                # Recover from lost gallery
                recovered = self._lost_gallery.pop(best_gid)
                recovered.last_seen = data['timestamp']
                recovered.last_x = data['world_x']
                recovered.last_y = data['world_y']
                recovered.cam_history.add(data['cam_id'])
                recovered.active_local_ids[data['cam_id']] = int(data['local_track_id'])
                recovered.last_cam_update[data['cam_id']] = data['timestamp']
                recovered.zone = self._get_zone(data['world_x'], data['world_y'])

                self._gallery[best_gid] = recovered
                self._cam_local_to_global[(data['cam_id'], int(data['local_track_id']))] = best_gid
                self.r.hset("global_id_map", f"{data['cam_id']}:{data['local_track_id']}", best_gid)
                log.info(f"Recovered global_id={best_gid} from lost gallery")
            else:
                still_unmatched.append(data)

        return still_unmatched

    def _cross_camera_dedup(self, active_in_batch):
        """Use Hungarian assignment to optimally match entries across cameras."""
        # Group gallery entries by their PRIMARY camera (most recent update)
        cam_groups = {}
        for gid, entry in self._gallery.items():
            primary_cam = max(entry.last_cam_update, key=entry.last_cam_update.get)
            if primary_cam not in cam_groups:
                cam_groups[primary_cam] = []
            cam_groups[primary_cam].append(gid)
        
        cam_list = list(cam_groups.keys())
        if len(cam_list) < 2:
            return
        
        merged = set()
        
        for ci in range(len(cam_list)):
            for cj in range(ci + 1, len(cam_list)):
                cam_a, cam_b = cam_list[ci], cam_list[cj]
                gids_a = [g for g in cam_groups[cam_a] if g not in merged]
                gids_b = [g for g in cam_groups[cam_b] if g not in merged]
                
                if not gids_a or not gids_b:
                    continue
                
                cost_matrix = np.full((len(gids_a), len(gids_b)), 1000.0, dtype=np.float32)
                
                for i, ga in enumerate(gids_a):
                    if ga not in self._gallery: continue
                    ea = self._gallery[ga]
                    for j, gb in enumerate(gids_b):
                        if gb not in self._gallery: continue
                        eb = self._gallery[gb]
                        
                        dist = self._distance_m(ea.last_x, ea.last_y, eb.last_x, eb.last_y)
                        if dist > self.dedup_spatial_radius_m:
                            continue
                        
                        # Check active conflict
                        conflict = False
                        shared = set(ea.active_local_ids.keys()) & set(eb.active_local_ids.keys())
                        for cam in shared:
                            id_a, id_b = ea.active_local_ids[cam], eb.active_local_ids[cam]
                            if id_a != id_b and id_a in active_in_batch.get(cam, set()) and id_b in active_in_batch.get(cam, set()):
                                conflict = True
                                break
                        if conflict:
                            continue
                        
                        has_feats = len(ea.feature_bank) > 0 and len(eb.feature_bank) > 0
                        if has_feats:
                            sim = self._bank_similarity(ea.feature_bank, eb.feature_bank)
                            if sim >= self.reid_sim_threshold:
                                cost_matrix[i, j] = dist * (1.0 - sim)
                            elif sim >= 0.40:
                                cost_matrix[i, j] = dist
                        else:
                            cost_matrix[i, j] = dist
                
                row_inds, col_inds = linear_sum_assignment(cost_matrix)
                
                for row, col in zip(row_inds, col_inds):
                    if cost_matrix[row, col] >= 1000.0:
                        continue
                    
                    ga, gb = gids_a[row], gids_b[col]
                    if ga not in self._gallery or gb not in self._gallery:
                        continue
                    
                    keep, drop = min(ga, gb), max(ga, gb)
                    ea_keep, ea_drop = self._gallery[keep], self._gallery[drop]
                    
                    ea_keep.feature_bank.extend(ea_drop.feature_bank)
                    ea_keep.feature_bank = ea_keep.feature_bank[-5:]
                    ea_keep.cam_history.update(ea_drop.cam_history)
                    
                    for cam_id, lid in ea_drop.active_local_ids.items():
                        if cam_id not in ea_keep.active_local_ids:
                            ea_keep.active_local_ids[cam_id] = lid
                            ea_keep.last_cam_update[cam_id] = ea_drop.last_cam_update.get(cam_id, 0)
                    
                    for k, v in list(self._cam_local_to_global.items()):
                        if v == drop:
                            self._cam_local_to_global[k] = keep
                            self.r.hset("global_id_map", f"{k[0]}:{k[1]}", keep)
                    
                    del self._gallery[drop]
                    merged.add(drop)
                    d = self._distance_m(ea_keep.last_x, ea_keep.last_y, ea_drop.last_x, ea_drop.last_y)
                    log.info(f"Dedup: Merged {drop} into {keep} (dist={d:.1f}m, cost={cost_matrix[row, col]:.2f})")

    def _create_new_id(self, data):
        gid, self._next_id = self._next_id, self._next_id + 1
        zone = self._get_zone(data['world_x'], data['world_y'])
        
        self._gallery[gid] = GalleryEntry(
            gid, 
            data['local_track_id'], 
            data['timestamp'], 
            data['world_x'], 
            data['world_y'], 
            data['cam_id'], 
            zone
        )
        self._cam_local_to_global[(data['cam_id'], int(data['local_track_id']))] = gid
        self.r.hset("global_id_map", f"{data['cam_id']}:{data['local_track_id']}", gid)
        log.info(f"New global_id={gid} in zone {zone}")

    def _update_entry(self, gid, data):
        e = self._gallery[gid]

        dt = data['timestamp'] - e.last_seen
        if dt > 0.1: 
            inst_vel_x = (data['world_x'] - e.last_x) / dt
            inst_vel_y = (data['world_y'] - e.last_y) / dt
            e.velocity_x = (e.velocity_x * 0.6) + (inst_vel_x * 0.4)
            e.velocity_y = (e.velocity_y * 0.6) + (inst_vel_y * 0.4)

        e.last_seen, e.last_x, e.last_y = data['timestamp'], data['world_x'], data['world_y']
        e.cam_history.add(data['cam_id'])
        e.zone = self._get_zone(data['world_x'], data['world_y'])
        
        e.active_local_ids[data['cam_id']] = int(data['local_track_id'])
        e.last_cam_update[data['cam_id']] = data['timestamp']

    def run(self):
        log.info("Starting Global Matcher (BEV + Spatial Priority)...")
        last_cleanup = time.time()
        
        while True:
            self._poll_reid_results()
            
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
                
            now = time.time()
            zone_counts = {z['name']: 0 for z in self.zones}
            zone_counts["Unknown"] = 0
            for e in self._gallery.values():
                if e.zone in zone_counts:
                    zone_counts[e.zone] += 1
                else:
                    zone_counts["Unknown"] += 1
                    
            snapshot = {
                "unique_people": len(self._gallery),
                "next_id": self._next_id,
                "entries": [
                    {
                        "global_id": e.global_id, 
                        "last_x": round(e.last_x, 2), 
                        "last_y": round(e.last_y, 2), 
                        "cameras": sorted(list(e.cam_history)),
                        "zone": e.zone
                    } for e in self._gallery.values()
                ],
                "zone_counts": zone_counts
            }
            self.r.set("state:gallery", json.dumps(snapshot))

if __name__ == "__main__":
    GlobalMatcher().run()