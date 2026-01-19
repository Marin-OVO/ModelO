import torch
import numpy as np
from typing import List, Tuple, Dict
from utils.lmds import LMDS


class MultiScaleCountCorrection:
    def __init__(
            self,
            lmds_kernel_size=3,
            lmds_adapt_ts=0.5,
            num_classes=2,
            count_threshold=2.0,
            scales=[1, 2, 4],  # 1 -> 1/1, 2 -> 1/2, 4 -> 1/4
            scale_weights=None,
    ):
        self.lmds = LMDS(
            kernel_size=(lmds_kernel_size, lmds_kernel_size),
            adapt_ts=lmds_adapt_ts
        )
        self.lmds_adapt_ts = lmds_adapt_ts
        self.num_classes = num_classes
        self.count_threshold = count_threshold
        self.scales = scales

        # init weight: [0.5, 0.3, 0.2]
        if scale_weights is None:
            self.scale_weights = {1: 0.5, 2: 0.3, 4: 0.2}
        else:
            self.scale_weights = scale_weights

    def correct_single_region(
            self,
            heatmap: torch.Tensor,      # (B, 2, H, W)
            density_map: torch.Tensor,  # (B, 1, H, W)
            region_slice: Tuple = None  # (h_start, h_end, w_start, w_end)
    ) -> Dict:
        """
            correct single region
        """
        if region_slice is not None:
            h_s, h_e, w_s, w_e = region_slice
            hm_region = heatmap[:, :, h_s:h_e, w_s:w_e]
            dm_region = density_map[:, :, h_s:h_e, w_s:w_e]
        else:
            hm_region = heatmap
            dm_region = density_map
            h_s, w_s = 0, 0

        # lmds
        counts, locs, labels, scores = self.lmds(hm_region)

        # density_map/lmds -> counts
        density_count = dm_region.sum().item()
        lmds_count = len(locs[0]) if locs and len(locs) > 0 else 0

        # diff
        count_diff = density_count - lmds_count

        corrected = False

        # |density_count - lmds_count| > th
        if abs(count_diff) > self.count_threshold:
            corrected = True

            # add
            if count_diff > 0:
                locs[0], labels[0], scores[0] = self._add_points(
                    hm_region, dm_region, locs[0], labels[0], scores[0],
                    num_to_add = int(round(count_diff * 0.5))
                )
            else:
                # remove
                num_to_remove = int(round(abs(count_diff)))
                if len(scores[0]) > num_to_remove:
                    sorted_indices = np.argsort(scores[0])[::-1]
                    keep_indices = sorted_indices[:-num_to_remove]

                    locs[0] = [locs[0][i] for i in keep_indices]
                    labels[0] = [labels[0][i] for i in keep_indices]
                    scores[0] = [scores[0][i] for i in keep_indices]

        if region_slice is not None and locs[0]:
            locs[0] = [(loc[0] + h_s, loc[1] + w_s) for loc in locs[0]]

        return {
            'locs': locs[0] if locs else [],
            'labels': labels[0] if labels else [],
            'scores': scores[0] if scores else [],
            'corrected': corrected,
            'density_count': density_count,
            'lmds_count': lmds_count
        }

    def _add_points(
            self,
            heatmap: torch.Tensor,
            density_map: torch.Tensor,
            locs: List,
            labels: List,
            scores: List,
            num_to_add: int
    ) -> Tuple[List, List, List]:
        """
            add point in high reaction region
        """
        if num_to_add <= 0:
            return locs, labels, scores

        # fg_heatmap if fg=1
        fg_heatmap = heatmap[0, 1].detach().cpu().numpy()

        mask = np.ones_like(fg_heatmap, dtype=bool)
        radius = 3

        for loc in locs:
            y, x = loc
            y, x = int(round(y)), int(round(x))
            y_min = max(0, y - radius)
            y_max = min(fg_heatmap.shape[0], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(fg_heatmap.shape[1], x + radius + 1)
            mask[y_min:y_max, x_min:x_max] = False

        masked_heatmap = fg_heatmap.copy()
        masked_heatmap[~mask] = -np.inf

        # find top num_to_add loc
        flat_indices = np.argpartition(masked_heatmap.ravel(), -num_to_add)[-num_to_add:]
        top_positions = np.unravel_index(flat_indices, fg_heatmap.shape)

        new_locs = list(locs)
        new_labels = list(labels)
        new_scores = list(scores)

        for i in range(len(top_positions[0])):
            y, x = top_positions[0][i], top_positions[1][i]
            if masked_heatmap[y, x] > self.lmds_adapt_ts - 0.05:
                new_locs.append((y, x))
                new_labels.append(1)  # labels: 1 = fg
                new_scores.append(float(masked_heatmap[y, x]))

        return new_locs, new_labels, new_scores

    def split_region(
            self,
            H: int,
            W: int,
            n_splits: int
    ) -> List[Tuple[int, int, int, int]]:
        """
            made images n_splits x n_splits

            Returns:
                List of (h_start, h_end, w_start, w_end)
        """
        regions = []
        h_step = H // n_splits
        w_step = W // n_splits

        for i in range(n_splits):
            for j in range(n_splits):
                h_s = i * h_step
                h_e = H if i == n_splits - 1 else (i + 1) * h_step
                w_s = j * w_step
                w_e = W if j == n_splits - 1 else (j + 1) * w_step
                regions.append((h_s, h_e, w_s, w_e))

        return regions

    def __call__(
            self,
            heatmap: torch.Tensor,
            density_map: torch.Tensor
    ) -> Dict:
        """
            main function

            Args:
                heatmap: (B, 2, H, W)
                density_map: (B, 1, H, W)
        """
        B, _, H, W = heatmap.shape

        # only val/test: batch size=1
        assert B == 1, "Currently only support batch_size=1"

        all_scale_results = []

        for scale in self.scales:
            if scale == 1:
                result = self.correct_single_region(heatmap, density_map)
                result['scale'] = 'global'
                all_scale_results.append(result)

            else:
                regions = self.split_region(H, W, scale)
                scale_locs = []
                scale_labels = []
                scale_scores = []

                for region in regions:
                    result = self.correct_single_region(
                        heatmap, density_map, region
                    )
                    scale_locs.extend(result['locs'])
                    scale_labels.extend(result['labels'])
                    scale_scores.extend(result['scores'])

                # remove too close
                scale_locs, scale_labels, scale_scores = self._deduplicate(
                    scale_locs, scale_labels, scale_scores
                )

                all_scale_results.append({
                    'locs': scale_locs,
                    'labels': scale_labels,
                    'scores': scale_scores,
                    'scale': f'{scale}x{scale}'
                })

        final_result = self._fuse_multi_scale(all_scale_results)

        return final_result

    def _deduplicate(
            self,
            locs: List,
            labels: List,
            scores: List,
            distance_threshold: int = 0.5
    ) -> Tuple[List, List, List]:
        """
            remove too close point
        """
        if not locs:
            return locs, labels, scores

        locs_arr = np.array(locs)
        scores_arr = np.array(scores)

        # sorted
        sorted_indices = np.argsort(scores_arr)[::-1]

        keep_indices = []
        for idx in sorted_indices:
            loc = locs_arr[idx]

            too_close = False
            for keep_idx in keep_indices:
                keep_loc = locs_arr[keep_idx]
                dist = np.sqrt(((loc - keep_loc) ** 2).sum())
                if dist < distance_threshold:
                    too_close = True
                    break

            if not too_close:
                keep_indices.append(idx)

        dedup_locs = [locs[i] for i in keep_indices]
        dedup_labels = [labels[i] for i in keep_indices]
        dedup_scores = [scores[i] for i in keep_indices]

        return dedup_locs, dedup_labels, dedup_scores

    def _fuse_multi_scale(
            self,
            all_scale_results: List[Dict]
    ) -> Dict:
        """
            fusion results
        """
        all_locs = []
        all_labels = []
        all_scores = []
        all_weights = []

        for result in all_scale_results:
            scale_str = result['scale']
            if scale_str == 'global':
                weight = self.scale_weights[1]
            elif scale_str == '2x2':
                weight = self.scale_weights[2]
            elif scale_str == '4x4':
                weight = self.scale_weights[4]
            else:
                weight = 0.1

            for loc, label, score in zip(result['locs'], result['labels'], result['scores']):
                all_locs.append(loc)
                all_labels.append(label)
                all_scores.append(score * weight)
                all_weights.append(weight)

        # cluster
        if not all_locs:
            return {
                'locs': [],
                'labels': [],
                'scores': []
            }

        final_locs, final_labels, final_scores = self._cluster_and_merge(
            all_locs, all_labels, all_scores, cluster_radius=1
        )

        return {
            'locs': final_locs,
            'labels': final_labels,
            'scores': final_scores
        }

    def _cluster_and_merge(
            self,
            locs: List,
            labels: List,
            scores: List,
            cluster_radius: int = 4
    ) -> Tuple[List, List, List]:
        """
            cluster and merge
        """
        if not locs:
            return [], [], []

        locs_arr = np.array(locs)
        labels_arr = np.array(labels)
        scores_arr = np.array(scores)

        used = np.zeros(len(locs), dtype=bool)
        clusters = []

        for i in range(len(locs)):
            if used[i]:
                continue

            distances = np.sqrt(((locs_arr - locs_arr[i]) ** 2).sum(axis=1))
            neighbors = np.where(distances < cluster_radius)[0]

            used[neighbors] = True

            cluster_locs = locs_arr[neighbors]
            cluster_scores = scores_arr[neighbors]
            cluster_labels = labels_arr[neighbors]

            weights = cluster_scores / cluster_scores.sum()
            merged_loc = (cluster_locs * weights[:, None]).sum(axis=0)

            merged_score = cluster_scores.max()

            merged_label = np.bincount(cluster_labels).argmax()

            clusters.append({
                'loc': tuple(merged_loc.astype(int)),
                'label': int(merged_label),
                'score': float(merged_score)
            })

        final_locs = [c['loc'] for c in clusters]
        final_labels = [c['label'] for c in clusters]
        final_scores = [c['score'] for c in clusters]

        return final_locs, final_labels, final_scores