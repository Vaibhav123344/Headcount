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

        # Confirmation gating: a fresh track is "tentative" and NOT counted until it
        # survives `confirm_hits` position updates. This lets cross-camera dedup merge
        # duplicates (person seen by 2 cams) BEFORE either copy is ever counted, and
        # discards 1-2 frame ghost detections. Kills the main headcount fluctuation.
        self.hits = 1
        self.confirmed = False

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
        self.spatial_match_radius_m = m_cfg.get('spatial_match_radius_m', 4.0)
        self.dedup_spatial_radius_m = m_cfg.get('dedup_spatial_radius_m', 2.0)
        self.feature_bank_size  = m_cfg.get('feature_bank_size', 5)
        self.recover_radius_m   = m_cfg.get('recover_radius_m', 3.0)
        self.stale_position_sec = m_cfg.get('stale_position_sec', 1.0)
        self.stale_reid_sec     = m_cfg.get('stale_reid_sec', 3.0)
        # Headcount stability knobs.
        self.confirm_hits       = m_cfg.get('confirm_hits', 3)      # updates before a track is counted
        self.count_hold_sec     = m_cfg.get('count_hold_sec', 3.0)  # keep counting after last_seen this long
        
        l_cfg = self.cfg.get('layout', {})
        self.pixels_per_meter = l_cfg.get('pixels_per_meter', 50.0)
        self.zones = l_cfg.get('zones', [])
        
        self._gallery = {}
        self._lost_gallery = {}
        self._cam_local_to_global = {}
        self._next_id = 1
        self._jitter_buffer = []  

    @staticmethod
    def _point_in_poly(x, y, pts):
        inside = False
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def _get_zone(self, x, y):
        for z in self.zones:
            if 'points' in z:
                if self._point_in_poly(x, y, z['points']):
                    return z['name']
            elif 'x1' in z:  # legacy rectangle format
                if z['x1'] <= x <= z['x2'] and z['y1'] <= y <= z['y2']:
                    return z['name']
        return "Unknown"

    @staticmethod
    def _normalize_keys(data):
        return {
            (k.decode('utf-8') if isinstance(k, bytes) else k): v
            for k, v in data.items()
        }

    def _drain(self, queue):
        """Atomically pull ALL items from a Redis list in one round-trip.

        LRANGE + DEL run inside a MULTI/EXEC transaction, so items pushed
        after the snapshot survive for the next drain (nothing is lost).
        Replaces slow per-item lpop loops.
        """
        pipe = self.r.pipeline()  # transaction=True by default
        pipe.lrange(queue, 0, -1)
        pipe.delete(queue)
        items, _ = pipe.execute()
        return items

    def _poll_reid_results(self):
        """Drain warehouse:queue:reid_result to update feature banks; skip stale embeddings."""
        now = time.time()
        for res in self._drain(self.reid_queue):
            data = self._normalize_keys(msgpack.unpackb(res, strict_map_key=False))
            # Drop embeddings that arrived too late to be trusted for this track's pose.
            if self.stale_reid_sec > 0 and 'timestamp' in data and (now - data['timestamp']) > self.stale_reid_sec:
                continue
            key = (data['cam_id'], int(data['local_track_id']))
            if key in self._cam_local_to_global:
                gid = self._cam_local_to_global[key]
                if gid in self._gallery and 'embedding' in data:
                    emb = np.array(data['embedding'], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0.01:
                        emb = emb / norm
                        
                        # Feature Bank Poisoning Protection
                        current_bank = self._gallery[gid].feature_bank
                        if len(current_bank) > 0:
                            sim = self._bank_similarity([emb], current_bank)
                            if sim < (self.reid_sim_threshold - 0.20): 
                                log.warning(f"Dropped poisoned ReID crop for {gid} (sim={sim:.2f})")
                                continue
                                
                        self._gallery[gid].feature_bank.append(emb)
                        if len(self._gallery[gid].feature_bank) > self.feature_bank_size:
                            self._gallery[gid].feature_bank.pop(0)

                        # Async ReID-based Lost Recovery for newly minted tracks
                        entry = self._gallery[gid]
                        if entry.hits < 30 and self._lost_gallery:
                            best_lost_gid = None
                            best_sim = 0.0
                            
                            for lost_gid, lost_entry in self._lost_gallery.items():
                                if len(lost_entry.feature_bank) > 0:
                                    sim = self._bank_similarity([emb], lost_entry.feature_bank)
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_lost_gid = lost_gid
                                        
                            if best_lost_gid and best_sim > (self.reid_sim_threshold + 0.10):
                                log.info(f"ReID RECOVERY: Track {gid} is actually lost {best_lost_gid} (sim={best_sim:.2f})")
                                
                                lost_entry = self._lost_gallery.pop(best_lost_gid)
                                lost_entry.last_seen = entry.last_seen
                                lost_entry.last_x = entry.last_x
                                lost_entry.last_y = entry.last_y
                                lost_entry.velocity_x = entry.velocity_x
                                lost_entry.velocity_y = entry.velocity_y
                                lost_entry.active_local_ids = entry.active_local_ids
                                lost_entry.last_cam_update = entry.last_cam_update
                                lost_entry.cam_history.update(entry.cam_history)
                                lost_entry.feature_bank.append(emb)
                                
                                self._gallery[best_lost_gid] = lost_entry
                                del self._gallery[gid]
                                
                                for k, v in list(self._cam_local_to_global.items()):
                                    if v == gid:
                                        self._cam_local_to_global[k] = best_lost_gid
                                        self.r.hset("global_id_map", f"{k[0]}:{k[1]}", best_lost_gid)

    def _gather_and_align_windows(self):
        for res in self._drain(self.in_queue):
            self._jitter_buffer.append(self._normalize_keys(msgpack.unpackb(res, strict_map_key=False)))

        if not self._jitter_buffer:
            res = self.r.blpop(self.in_queue, timeout=1)
            if res: self._jitter_buffer.append(self._normalize_keys(msgpack.unpackb(res[1], strict_map_key=False)))

        if not self._jitter_buffer:
            return []

        self._jitter_buffer.sort(key=lambda x: x['timestamp'])
        latest_stream_time = self._jitter_buffer[-1]['timestamp']

        # Self-healing latency guard: if we ever fall far behind, discard ancient
        # backlog positions so we always process near-live data (never live data).
        if self.stale_position_sec > 0:
            cutoff = latest_stream_time - self.stale_position_sec
            before = len(self._jitter_buffer)
            self._jitter_buffer = [d for d in self._jitter_buffer if d['timestamp'] >= cutoff]
            dropped = before - len(self._jitter_buffer)
            if dropped:
                log.warning(f"Dropped {dropped} stale positions (>{self.stale_position_sec}s behind); matcher was backlogged")

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

                # Calculate expected position based on velocity
                expected_x = entry.last_x + (entry.velocity_x * dt)
                expected_y = entry.last_y + (entry.velocity_y * dt)
                dist_to_expected = self._distance_m(world_x, world_y, expected_x, expected_y)
                
                # Blended cost (70% actual distance, 30% expected trajectory)
                blended_cost = (dist * 0.7) + (dist_to_expected * 0.3)

                # Strict distance limit to prevent random jumping
                if dist <= self.spatial_match_radius_m:
                    cost_matrix[i, j] = blended_cost

        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        assigned_rows = set()

        for row, col in zip(row_inds, col_inds):
            if cost_matrix[row, col] <= self.spatial_match_radius_m:
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

                # Must be within reasonable distance to recover.
                # Recovery is spatial-only: the fresh local track has no feature bank yet,
                # and fast-path payloads carry no embedding (ReID arrives async, later).
                if dist > self.recover_radius_m:
                    continue

                if dist < best_score:
                    best_score = dist
                    best_gid = gid

            if best_gid is not None and best_score < self.recover_radius_m:
                # Recover from lost gallery
                recovered = self._lost_gallery.pop(best_gid)
                recovered.last_seen = data['timestamp']
                recovered.last_x = data['world_x']
                recovered.last_y = data['world_y']
                recovered.cam_history.add(data['cam_id'])
                recovered.active_local_ids[data['cam_id']] = int(data['local_track_id'])
                recovered.last_cam_update[data['cam_id']] = data['timestamp']
                recovered.zone = self._get_zone(data['world_x'], data['world_y'])
                recovered.hits += 1  # already-known person keeps confirmed state on recovery

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
                                continue # They look different, reject merge
                        else:
                            # No ReID yet. Only merge if extremely close.
                            if dist <= 1.0:
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
                    ea_keep.feature_bank = ea_keep.feature_bank[-self.feature_bank_size:]
                    ea_keep.cam_history.update(ea_drop.cam_history)
                    # Inherit confirmation so a merge never demotes a counted person.
                    ea_keep.confirmed = ea_keep.confirmed or ea_drop.confirmed
                    ea_keep.hits = max(ea_keep.hits, ea_drop.hits)
                    
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

        e.hits += 1
        if e.hits >= self.confirm_hits:
            e.confirmed = True

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

                # Prune stale per-camera bindings so a camera that lost this person
                # long ago can't falsely block re-binding / cross-camera matching later.
                for e in self._gallery.values():
                    for c in [c for c, t in e.last_cam_update.items() if now - t > self.lost_short_sec]:
                        e.active_local_ids.pop(c, None)
                        e.last_cam_update.pop(c, None)
                        e.cam_history.discard(c)
                last_cleanup = now
                
            windows = self._gather_and_align_windows()
            for window in windows:
                self.process_window(window)
                
            now = time.time()
            # Count only CONFIRMED tracks that were seen recently. Confirmation drops
            # ghosts + pre-dedup duplicates; the hold window keeps the count steady
            # through brief detection gaps instead of flickering frame-to-frame.
            counted = [
                e for e in self._gallery.values()
                if e.confirmed and (now - e.last_seen) <= self.count_hold_sec
            ]

            zone_counts = {z['name']: 0 for z in self.zones}
            zone_counts["Unknown"] = 0
            for e in counted:
                if e.zone in zone_counts:
                    zone_counts[e.zone] += 1
                else:
                    zone_counts["Unknown"] += 1

            snapshot = {
                "unique_people": len(counted),
                "next_id": self._next_id,
                "entries": [
                    {
                        "global_id": e.global_id,
                        "last_x": round(e.last_x, 2),
                        "last_y": round(e.last_y, 2),
                        "cameras": sorted(list(e.cam_history)),
                        "zone": e.zone
                    } for e in counted
                ],
                "zone_counts": zone_counts
            }
            self.r.set("state:gallery", json.dumps(snapshot))

if __name__ == "__main__":
    GlobalMatcher().run()