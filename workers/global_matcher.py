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
        self.embedding   = embedding
        self.last_seen   = last_seen
        self.last_x      = last_x
        self.last_y      = last_y
        self.velocity_x  = velocity_x    # ── Upgrade 2: velocity for time-alignment
        self.velocity_y  = velocity_y
        self.cam_history = {cam_id}

class GlobalMatcher:
    """
    Multi-stage global matcher with:
      - Stage 0: Identity lock (pre-matched tracks bypass matching)
      - Stage 1: High-confidence Hungarian (strict visual + spatial)
      - Stage 2: DBSCAN spatial-density backup (relaxed visual, spatial clustering)
      - Stage 3: New global ID creation
    """
    # ── Matching thresholds ──
    STAGE1_SIM_THRESH   = 0.75   # cosine similarity ≥ this for Stage 1 candidacy
    STAGE1_DIST_THRESH  = 1.5    # meters – max spatial distance for Stage 1
    STAGE1_COST_ACCEPT  = 0.25   # accept Stage 1 matches below this fused cost
    STAGE2_SIM_THRESH   = 0.50   # relaxed cosine similarity for DBSCAN stage
    DBSCAN_EPS          = 1.5    # meters – DBSCAN neighbourhood radius
    DBSCAN_MIN_SAMPLES  = 2      # minimum cluster size

    # ── Cost weighting ──
    ALPHA_NORMAL        = 0.3    # weight for visual cost (normal visibility)
    ALPHA_OCCLUDED      = 0.8    # weight for visual cost (occluded – spatial unreliable)

    def __init__(self, similarity_threshold=0.62, ema_alpha=0.1,
                 expiry_sec=10.0, max_speed_mps=5.0):
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        self.in_queue = "warehouse:queue:matcher"
        
        self.sim_thresh   = similarity_threshold
        self.ema          = ema_alpha
        self.expiry       = expiry_sec
        self.max_speed_mps = max_speed_mps
        
        self._gallery              = {}   # global_id -> GalleryEntry
        self._cam_local_to_global  = {}   # (cam_id, local_id) -> global_id
        self._next_id              = 1
        self._expected_dim         = None # set from first embedding seen

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
    # Upgrade 2: Time-aligned spatial distance
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
            
        # If gallery is empty, everything is new
        if not self._gallery:
            for data in unmatched_items:
                self._create_new_id(data)
            return

        # ── Stage 1: High-Confidence Hungarian Matching ──
        still_unmatched = self._stage1_hungarian(unmatched_items)

        # ── Stage 2: DBSCAN Spatial-Density Backup ──
        still_unmatched = self._stage2_dbscan(still_unmatched)

        # ── Stage 3: Create new IDs for anything left ──
        for data in still_unmatched:
            self._create_new_id(data)

    # ─────────────────────────────────────────────────────────────────
    # Stage 1: High-confidence Hungarian
    # ─────────────────────────────────────────────────────────────────
    def _stage1_hungarian(self, items):
        """
        Match tracks where BOTH visual cosine similarity AND spatial distance
        are exceptionally strong. Uses the Hungarian algorithm for optimal 1:1
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
                
                # ── Upgrade 2: Time-aligned spatial distance ──
                spatial_dist = self._time_aligned_distance(data, entry)

                # Spatial gate: reject if implied speed is too fast
                time_diff = max(0.01, data['timestamp'] - entry.last_seen)
                speed = spatial_dist / time_diff
                if speed > self.max_speed_mps:
                    continue  # cost stays 1000.0
                
                # Stage 1 strict gates
                if spatial_dist > self.STAGE1_DIST_THRESH:
                    continue  # too far for high-confidence match

                # Cosine similarity
                sim = float(np.dot(entry.embedding, data['features_norm']))
                if sim < self.STAGE1_SIM_THRESH:
                    continue  # visual match not strong enough for Stage 1

                cost_visual  = 1.0 - sim
                cost_spatial = min(1.0, spatial_dist / self.STAGE1_DIST_THRESH)

                # ── Upgrade 5: Occlusion-aware cost weighting ──
                alpha = self.ALPHA_OCCLUDED if data.get('occluded', False) else self.ALPHA_NORMAL
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
        This handles cases where OSNet embeddings are noisy (back turned,
        motion blur) but the person is clearly occupying the same area.
        """
        if not items or not self._gallery:
            return items

        # Collect unmatched gallery entries (those not recently matched in Stage 1)
        # We consider ALL active gallery entries as candidates
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
                    sim   = float(np.dot(entry.embedding, data['features_norm']))
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
        """Update a gallery entry with new observation (EMA embedding + velocity)."""
        e = self._gallery[gid]
        
        # Smoothen appearance embedding using EMA (prevents sudden lighting noise)
        new_emb = (1.0 - self.ema) * e.embedding + self.ema * emb
        nn = np.linalg.norm(new_emb)
        if nn > 0:
            new_emb = new_emb / nn
            
        e.embedding   = new_emb
        e.last_seen   = data['timestamp']
        e.last_x      = data['world_x']
        e.last_y      = data['world_y']
        e.velocity_x  = data.get('velocity_x', 0.0)
        e.velocity_y  = data.get('velocity_y', 0.0)
        e.cam_history.add(cam_id)

    # ─────────────────────────────────────────────────────────────────
    # Track expiry
    # ─────────────────────────────────────────────────────────────────
    def cleanup(self):
        now = time.time()
        # Expire tracks that haven't been seen in expiry_sec
        stale = [gid for gid, e in self._gallery.items() if now - e.last_seen > self.expiry]
        for gid in stale:
            log.info(f"Expiring global_id={gid} due to inactivity.")
            del self._gallery[gid]
            for k, v in list(self._cam_local_to_global.items()):
                if v == gid:
                    cam_id, local_id = k
                    self.r.hdel("global_id_map", f"{cam_id}:{local_id}")
                    del self._cam_local_to_global[k]

    # ─────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────
    def run(self):
        log.info("Starting Multi-Stage Global Matcher (Hungarian + DBSCAN + Velocity Alignment)...")
        last_cleanup = time.time()
        
        while True:
            # Periodically cleanup expired tracks
            if time.time() - last_cleanup > 5.0:
                self.cleanup()
                last_cleanup = time.time()
                
            batch = self._gather_batch(max_batch_size=32)
            if batch:
                self.process_batch(batch)
                
                # ── Upgrade 6: Enriched gallery snapshot for dashboard ──
                now = time.time()
                snapshot = {
                    "unique_people": len(self._gallery),
                    "next_id": self._next_id,
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
