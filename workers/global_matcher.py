import redis
import json
import numpy as np
import time
import math
import logging
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("global_matcher")

class GalleryEntry:
    """Represents a single globally-tracked person in the gallery."""
    def __init__(self, global_id, embedding, last_seen, last_x, last_y, cam_id,
                 velocity_x=0.0, velocity_y=0.0):
        self.global_id  = global_id
        self.feature_bank = [embedding]  # Store up to 5 high-quality embeddings
        self.last_seen   = last_seen
        self.last_x      = last_x
        self.last_y      = last_y
        self.velocity_x  = velocity_x
        self.velocity_y  = velocity_y
        self.cam_history = {cam_id}

class GlobalMatcher:
    """
    Multi-stage global matcher with:
      - Stage 0: Identity lock (pre-matched tracks bypass matching)
      - Stage 1: High-confidence Hungarian (visual + spatial, relaxed for cross-cam)
      - Stage 2: DBSCAN spatial-density backup (relaxed visual, spatial clustering)
      - Stage 2.5: Lost Gallery search (re-identify expired tracks before creating new IDs)
      - Stage 3: New global ID creation + immediate cross-camera dedup
    """
    # ── Matching thresholds (relaxed for robust cross-camera matching) ──
    STAGE1_SIM_THRESH   = 0.60   # cosine similarity ≥ this for Stage 1 candidacy
    STAGE1_DIST_THRESH  = 3.0    # meters – max spatial distance for Stage 1
    STAGE1_COST_ACCEPT  = 0.35   # accept Stage 1 matches below this fused cost
    STAGE2_SIM_THRESH   = 0.45   # relaxed cosine similarity for DBSCAN stage
    DBSCAN_EPS          = 3.0    # meters – DBSCAN neighbourhood radius
    DBSCAN_MIN_SAMPLES  = 2      # minimum cluster size

    # ── Lost Gallery (re-identification of expired tracks) ──
    LOST_SIM_THRESH     = 0.55   # minimum similarity to reclaim a lost ID
    LOST_EXPIRY_SEC     = 120.0  # keep lost entries for 2 minutes

    # ── Cross-camera dedup ──
    DEDUP_SIM_THRESH    = 0.55   # merge new IDs from different cameras above this

    def __init__(self, similarity_threshold=0.62,
                 expiry_sec=30.0, max_speed_mps=5.0):
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        self.in_queue = "warehouse:queue:matcher"
        
        self.sim_thresh    = similarity_threshold
        self.expiry        = expiry_sec
        self.max_speed_mps = max_speed_mps
        
        self._gallery              = {}   # global_id -> GalleryEntry
        self._lost_gallery         = {}   # global_id -> GalleryEntry (expired, awaiting re-ID)
        self._cam_local_to_global  = {}   # (cam_id, local_id) -> global_id
        self._next_id              = 1
        self._expected_dim         = None # set from first embedding seen

    # ─────────────────────────────────────────────────────────────────
    # Feature Bank similarity helper
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _bank_similarity(feature_bank, query_emb):
        """Return the maximum cosine similarity between query and all bank entries."""
        return max(float(np.dot(emb, query_emb)) for emb in feature_bank)

    # ─────────────────────────────────────────────────────────────────
    # Batch gathering
    # ─────────────────────────────────────────────────────────────────
    def _gather_batch(self, max_batch_size=32):
        batch = []
        # Block for a very short time waiting for the first element
        res = self.r.blpop(self.in_queue, timeout=1)
        if not res:
            return batch
            
        batch.append(json.loads(res[1].decode('utf-8')))
        
        # Non-blocking pop to instantly grab everything else currently in the queue
        while len(batch) < max_batch_size:
            res = self.r.lpop(self.in_queue)
            if res:
                batch.append(json.loads(res.decode('utf-8')))
            else:
                break  # No more items, process immediately!
                
        return batch

    # ─────────────────────────────────────────────────────────────────
    # Time-aligned spatial distance
    # ─────────────────────────────────────────────────────────────────
    def _time_aligned_distance(self, data, entry):
        """
        Project the gallery entry's last position forward in time using its
        velocity vector, then compute Euclidean distance to the detection.
        This compensates for async camera timestamps.
        """
        ref_time  = data['timestamp']
        time_diff = ref_time - entry.last_seen

        # Project gallery entry forward using its velocity
        projected_x = entry.last_x + entry.velocity_x * time_diff
        projected_y = entry.last_y + entry.velocity_y * time_diff

        return math.hypot(data['world_x'] - projected_x,
                          data['world_y'] - projected_y)

    # ─────────────────────────────────────────────────────────────────
    # Main batch processing – Multi-stage matching
    # ─────────────────────────────────────────────────────────────────
    def process_batch(self, batch):
        unmatched_items = []
        
        for data in batch:
            emb = np.array(data['embedding'], dtype=np.float32)
            # Normalize embedding
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            data['features_norm'] = emb
            
            # Set expected dimension from first embedding seen
            if self._expected_dim is None:
                self._expected_dim = emb.shape[0]
                log.info(f"Feature dimension set to {self._expected_dim}")
            
            # Skip items with wrong feature dimensions
            if emb.shape[0] != self._expected_dim:
                log.warning(f"Skipping item with dim {emb.shape[0]} (expected {self._expected_dim})")
                continue
            
            cam_id   = data['cam_id']
            local_id = int(data['local_track_id'])
            key      = (cam_id, local_id)
            
            # ── Stage 0: Identity Lock ──
            # Pre-locked tracks bypass Hungarian matching entirely
            if key in self._cam_local_to_global and self._cam_local_to_global[key] in self._gallery:
                gid = self._cam_local_to_global[key]
                self._update_entry(gid, emb, data, cam_id)
                self.r.hset("global_id_map", f"{cam_id}:{local_id}", gid)
            else:
                unmatched_items.append(data)
                
        if not unmatched_items:
            return
            
        # If gallery is empty, everything is new — then immediately deduplicate
        if not self._gallery:
            for data in unmatched_items:
                self._create_new_id(data)
            self._cross_camera_dedup()
            return

        # ── Stage 1: High-Confidence Hungarian Matching ──
        still_unmatched = self._stage1_hungarian(unmatched_items)

        # ── Stage 2: DBSCAN Spatial-Density Backup ──
        still_unmatched = self._stage2_dbscan(still_unmatched)

        # ── Stage 2.5: Search Lost Gallery before creating new IDs ──
        still_unmatched = self._stage_lost_gallery(still_unmatched)

        # ── Stage 3: Create new IDs for anything left ──
        for data in still_unmatched:
            self._create_new_id(data)

        # ── Post-processing: Merge duplicate cross-camera IDs ──
        if still_unmatched:
            self._cross_camera_dedup()

    # ─────────────────────────────────────────────────────────────────
    # Stage 1: High-confidence Hungarian
    # ─────────────────────────────────────────────────────────────────
    def _stage1_hungarian(self, items):
        """
        Match tracks where BOTH visual cosine similarity AND spatial distance
        are strong enough. Uses the Hungarian algorithm for optimal 1:1
        assignment.
        """
        gallery_gids  = list(self._gallery.keys())
        num_items     = len(items)
        num_gallery   = len(gallery_gids)

        # Initialize cost matrix at 1000.0 (unmatchable)
        cost_matrix = np.full((num_items, num_gallery), 1000.0, dtype=np.float32)
        
        for i, data in enumerate(items):
            for j, gid in enumerate(gallery_gids):
                entry = self._gallery[gid]
                
                # Time-aligned spatial distance
                spatial_dist = self._time_aligned_distance(data, entry)

                # Spatio-Temporal Constraint: use a 0.5s grace period to prevent
                # minor async timestamps from creating infinite fake speeds
                time_diff_safe = max(0.5, data['timestamp'] - entry.last_seen)
                speed_safe = spatial_dist / time_diff_safe
                
                if speed_safe > self.max_speed_mps:
                    continue  # physically impossible movement -> cost stays 1000.0
                
                # Spatial gate
                if spatial_dist > self.STAGE1_DIST_THRESH:
                    continue  # too far

                # Feature Bank: max cosine similarity across all stored embeddings
                sim = self._bank_similarity(entry.feature_bank, data['features_norm'])
                
                if sim < self.STAGE1_SIM_THRESH:
                    continue  # visual match not strong enough

                cost_visual  = 1.0 - sim
                cost_spatial = min(1.0, spatial_dist / self.STAGE1_DIST_THRESH)

                # Adaptive Weighting based on keypoint quality
                kp_quality = data.get('keypoint_quality', 'high')
                if data.get('occluded', False) or kp_quality == 'low':
                    alpha = 0.8   # Trust visual more when spatial is unreliable
                else:
                    alpha = 0.6   # Default: 60% Visual, 40% Spatial
                    
                cost_matrix[i, j] = alpha * cost_visual + (1.0 - alpha) * cost_spatial
                
        # Solve Hungarian
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        
        assigned_rows = set()
        for row, col in zip(row_inds, col_inds):
            cost = cost_matrix[row, col]
            if cost < self.STAGE1_COST_ACCEPT:
                data     = items[row]
                gid      = gallery_gids[col]
                cam_id   = data['cam_id']
                local_id = int(data['local_track_id'])
                
                log.info(f"Stage1 Match: global_id={gid} (cam={cam_id} local={local_id} cost={cost:.3f})")
                
                self._cam_local_to_global[(cam_id, local_id)] = gid
                self._update_entry(gid, data['features_norm'], data, cam_id)
                self.r.hset("global_id_map", f"{cam_id}:{local_id}", gid)
                assigned_rows.add(row)

        # Return items that were NOT matched in Stage 1
        return [items[i] for i in range(len(items)) if i not in assigned_rows]

    # ─────────────────────────────────────────────────────────────────
    # Stage 2: DBSCAN spatial-density backup
    # ─────────────────────────────────────────────────────────────────
    def _stage2_dbscan(self, items):
        """
        For remaining unmatched tracks, use DBSCAN to cluster by spatial
        proximity. Within each cluster, match by best cosine similarity.
        """
        if not items or not self._gallery:
            return items

        gallery_entries = list(self._gallery.values())
        gallery_gids    = [e.global_id for e in gallery_entries]

        # Build a combined coordinate array: [detections..., gallery_entries...]
        det_coords = np.array([[d['world_x'], d['world_y']] for d in items])
        gal_coords = np.array([[e.last_x, e.last_y] for e in gallery_entries])
        all_coords = np.vstack([det_coords, gal_coords])

        n_det = len(items)
        n_gal = len(gallery_entries)

        # Run DBSCAN on spatial coordinates
        clustering = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MIN_SAMPLES,
                            metric='euclidean').fit(all_coords)
        labels = clustering.labels_

        assigned_rows = set()

        # Process each cluster
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # noise points – skip

            # Indices in the combined array belonging to this cluster
            cluster_mask   = (labels == label)
            det_in_cluster = [i for i in range(n_det) if cluster_mask[i]]
            gal_in_cluster = [i - n_det for i in range(n_det, n_det + n_gal) if cluster_mask[i]]

            if not det_in_cluster or not gal_in_cluster:
                continue  # need at least one detection AND one gallery entry

            # Within cluster: find best cosine-similarity match for each detection
            for di in det_in_cluster:
                if di in assigned_rows:
                    continue
                data      = items[di]
                best_sim  = -1.0
                best_gidx = -1

                for gi in gal_in_cluster:
                    entry = gallery_entries[gi]
                    sim = self._bank_similarity(entry.feature_bank, data['features_norm'])
                    if sim > best_sim:
                        best_sim  = sim
                        best_gidx = gi

                if best_sim >= self.STAGE2_SIM_THRESH and best_gidx >= 0:
                    entry    = gallery_entries[best_gidx]
                    gid      = entry.global_id
                    cam_id   = data['cam_id']
                    local_id = int(data['local_track_id'])

                    log.info(f"Stage2 DBSCAN Match: global_id={gid} "
                             f"(cam={cam_id} local={local_id} sim={best_sim:.3f})")

                    self._cam_local_to_global[(cam_id, local_id)] = gid
                    self._update_entry(gid, data['features_norm'], data, cam_id)
                    self.r.hset("global_id_map", f"{cam_id}:{local_id}", gid)
                    assigned_rows.add(di)

        # Return items that were NOT matched in Stage 2
        return [items[i] for i in range(len(items)) if i not in assigned_rows]

    # ─────────────────────────────────────────────────────────────────
    # Stage 2.5: Lost Gallery Search (re-identify expired tracks)
    # ─────────────────────────────────────────────────────────────────
    def _stage_lost_gallery(self, items):
        """
        Before creating a brand-new global ID, check if this person was
        recently tracked but expired. If their visual appearance matches
        a lost entry, reclaim the original global ID instead of creating
        a new one. This prevents ID fragmentation when people briefly
        leave the frame or switch between cameras.
        """
        if not items or not self._lost_gallery:
            return items

        still_unmatched = []
        for data in items:
            best_sim = -1.0
            best_gid = None

            for gid, entry in self._lost_gallery.items():
                sim = self._bank_similarity(entry.feature_bank, data['features_norm'])
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid

            if best_sim >= self.LOST_SIM_THRESH and best_gid is not None:
                # Reclaim the lost ID — move it back to active gallery
                entry = self._lost_gallery.pop(best_gid)
                cam_id   = data['cam_id']
                local_id = int(data['local_track_id'])

                log.info(f"Reclaimed lost global_id={best_gid} "
                         f"(cam={cam_id} local={local_id} sim={best_sim:.3f})")

                # Re-activate the entry
                entry.last_seen  = data['timestamp']
                entry.last_x     = data['world_x']
                entry.last_y     = data['world_y']
                entry.velocity_x = data.get('velocity_x', 0.0)
                entry.velocity_y = data.get('velocity_y', 0.0)
                entry.feature_bank.append(data['features_norm'])
                if len(entry.feature_bank) > 5:
                    entry.feature_bank.pop(0)
                entry.cam_history.add(cam_id)

                self._gallery[best_gid] = entry
                self._cam_local_to_global[(cam_id, local_id)] = best_gid
                self.r.hset("global_id_map", f"{cam_id}:{local_id}", best_gid)
            else:
                still_unmatched.append(data)

        return still_unmatched

    # ─────────────────────────────────────────────────────────────────
    # Cross-camera deduplication
    # ─────────────────────────────────────────────────────────────────
    def _cross_camera_dedup(self):
        """
        After creating new IDs, check if any two IDs from DIFFERENT cameras
        are visually the same person. If so, merge them into one ID.
        This prevents the initial burst of duplicate IDs when the system starts.
        """
        gids = list(self._gallery.keys())
        if len(gids) < 2:
            return

        merged = set()
        for i in range(len(gids)):
            if gids[i] in merged:
                continue
            entry_a = self._gallery.get(gids[i])
            if entry_a is None:
                continue

            for j in range(i + 1, len(gids)):
                if gids[j] in merged:
                    continue
                entry_b = self._gallery.get(gids[j])
                if entry_b is None:
                    continue

                # Only merge across different cameras
                if entry_a.cam_history == entry_b.cam_history:
                    continue

                # Check visual similarity between their feature banks
                best_sim = -1.0
                for emb_a in entry_a.feature_bank:
                    for emb_b in entry_b.feature_bank:
                        sim = float(np.dot(emb_a, emb_b))
                        if sim > best_sim:
                            best_sim = sim

                if best_sim >= self.DEDUP_SIM_THRESH:
                    # Merge entry_b into entry_a (keep the lower ID)
                    keep_gid = min(gids[i], gids[j])
                    drop_gid = max(gids[i], gids[j])

                    keep_entry = self._gallery[keep_gid]
                    drop_entry = self._gallery[drop_gid]

                    # Merge feature banks
                    for emb in drop_entry.feature_bank:
                        keep_entry.feature_bank.append(emb)
                    if len(keep_entry.feature_bank) > 5:
                        keep_entry.feature_bank = keep_entry.feature_bank[-5:]

                    # Merge camera history
                    keep_entry.cam_history.update(drop_entry.cam_history)

                    # Update all local→global mappings that pointed to drop_gid
                    for k, v in list(self._cam_local_to_global.items()):
                        if v == drop_gid:
                            self._cam_local_to_global[k] = keep_gid
                            cam_id, local_id = k
                            self.r.hset("global_id_map", f"{cam_id}:{local_id}", keep_gid)

                    # Remove the duplicate
                    del self._gallery[drop_gid]
                    merged.add(drop_gid)

                    log.info(f"Dedup: merged global_id={drop_gid} into global_id={keep_gid} "
                             f"(sim={best_sim:.3f})")

    # ─────────────────────────────────────────────────────────────────
    # ID creation and entry updates
    # ─────────────────────────────────────────────────────────────────
    def _create_new_id(self, data):
        gid      = self._next_id
        self._next_id += 1
        cam_id   = data['cam_id']
        local_id = int(data['local_track_id'])
        
        self._gallery[gid] = GalleryEntry(
            gid, data['features_norm'], data['timestamp'],
            data['world_x'], data['world_y'], cam_id,
            velocity_x=data.get('velocity_x', 0.0),
            velocity_y=data.get('velocity_y', 0.0)
        )
        self._cam_local_to_global[(cam_id, local_id)] = gid
        self.r.hset("global_id_map", f"{cam_id}:{local_id}", gid)
        
        log.info(f"New global_id={gid} (cam={cam_id} local={local_id})")

    def _update_entry(self, gid, emb, data, cam_id):
        """Update a gallery entry with new observation (Feature Bank + velocity)."""
        e = self._gallery[gid]
        
        # Append to Feature Bank (max 5)
        e.feature_bank.append(emb)
        if len(e.feature_bank) > 5:
            e.feature_bank.pop(0)  # Remove oldest
            
        e.last_seen   = data['timestamp']
        e.last_x      = data['world_x']
        e.last_y      = data['world_y']
        e.velocity_x  = data.get('velocity_x', 0.0)
        e.velocity_y  = data.get('velocity_y', 0.0)
        e.cam_history.add(cam_id)

    # ─────────────────────────────────────────────────────────────────
    # Track expiry (move to Lost Gallery instead of deleting)
    # ─────────────────────────────────────────────────────────────────
    def cleanup(self):
        now = time.time()

        # Move stale active tracks to the Lost Gallery (instead of deleting)
        stale = [gid for gid, e in self._gallery.items() if now - e.last_seen > self.expiry]
        for gid in stale:
            log.info(f"Expiring global_id={gid} → moved to lost gallery for re-identification.")
            entry = self._gallery.pop(gid)
            self._lost_gallery[gid] = entry  # Keep for re-identification

            # Remove the cam_local→global mapping so Stage 0 doesn't try to use the expired entry
            for k, v in list(self._cam_local_to_global.items()):
                if v == gid:
                    cam_id, local_id = k
                    self.r.hdel("global_id_map", f"{cam_id}:{local_id}")
                    del self._cam_local_to_global[k]

        # Permanently delete truly old lost entries
        lost_stale = [gid for gid, e in self._lost_gallery.items()
                      if now - e.last_seen > self.LOST_EXPIRY_SEC]
        for gid in lost_stale:
            del self._lost_gallery[gid]

    # ─────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────
    def run(self):
        log.info("Starting Multi-Stage Global Matcher (Feature Bank + Lost Gallery + Dedup)...")
        last_cleanup = time.time()
        
        while True:
            # Periodically cleanup expired tracks
            if time.time() - last_cleanup > 5.0:
                self.cleanup()
                last_cleanup = time.time()
                
            batch = self._gather_batch(max_batch_size=32)
            if batch:
                self.process_batch(batch)
                
                # Enriched gallery snapshot for dashboard
                now = time.time()
                snapshot = {
                    "unique_people": len(self._gallery),
                    "next_id": self._next_id,
                    "lost_count": len(self._lost_gallery),
                    "entries": [{
                        "global_id":  e.global_id,
                        "last_x":     round(e.last_x, 2),
                        "last_y":     round(e.last_y, 2),
                        "velocity_x": round(e.velocity_x, 2),
                        "velocity_y": round(e.velocity_y, 2),
                        "age_sec":    round(now - e.last_seen, 1),
                        "cameras":    sorted(e.cam_history),
                    } for e in self._gallery.values()]
                }
                self.r.set("state:gallery", json.dumps(snapshot))

if __name__ == "__main__":
    matcher = GlobalMatcher()
    matcher.run()
