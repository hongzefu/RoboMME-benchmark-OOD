import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import cv2
import copy


def process_segmentation(
    segmentation: np.ndarray,
    segmentation_id_map: Optional[Dict[int, Any]],
    color_map: Dict[int, List[int]],
    current_segment: Any,
    current_subgoal_segment: Optional[str],
    previous_subgoal_segment: Optional[str],
    current_task_name: str,
    existing_points: Optional[List[List[int]]] = None,
    existing_subgoal_filled: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Shared helper to compute segmentation filtering and grounded subgoal text.

    Returns a dict with:
    - segmentation_result: segmentation mask filtered to visible ids
    - segmentation_result_2d: squeezed version of segmentation_result
    - segmentation_points: cached center points for current targets
    - current_subgoal_segment_filled: subgoal string with centers filled in
    - no_object_flag: whether the target ids are missing in the mask
    - updated_previous_subgoal_segment: equals current_subgoal_segment for caller caching
    - vis_obj_id_list: ids kept in segmentation_result
    """
    segmentation_2d = segmentation.squeeze() if segmentation.ndim > 2 else segmentation

    if isinstance(current_segment, (list, tuple)):
        active_segments = list(current_segment)
    elif current_segment is None:
        active_segments = []
    else:
        active_segments = [current_segment]

    segment_ids_by_index = {idx: [] for idx in range(len(active_segments))}
    vis_obj_id_list: List[int] = []
    if isinstance(segmentation_id_map, dict):
        for obj_id, obj in sorted(segmentation_id_map.items()):
            if active_segments:
                for idx, target in enumerate(active_segments):
                    if obj is target:
                        vis_obj_id_list.append(obj_id)
                        segment_ids_by_index[idx].append(obj_id)
                        break
            if getattr(obj, "name", None) == "table-workspace":
                color_map[obj_id] = [0, 0, 0]

    segmentation_result = np.where(
        np.isin(segmentation_2d, vis_obj_id_list), segmentation_2d, 0
    )
    segmentation_result_2d = segmentation_result.squeeze()

    segmentation_points = existing_points or []
    current_subgoal_segment_filled = existing_subgoal_filled
    no_object_flag = False

    # V6 审查修复 F4（K5，只对打了 ``_robomme_refresh_on_move_px`` 标的目标 actor 生效）：
    # 子目标未切换、缓存中心已存在、且任一目标当前分割中心相对缓存的切比雪夫距离超过阈值时，
    # 按「切换」路径整体重算中心并重新填充文本（交换后的容器不再沿用旧坐标）。
    # 未打标的 actor 一律不进这段，原三档与其他环境逐字不变；目标不可见（缺分割）时不刷新（遮挡回填不在本轮范围）。
    refresh_on_move = False
    if current_subgoal_segment == previous_subgoal_segment and existing_points and active_segments:
        thresholds = [getattr(target, "_robomme_refresh_on_move_px", 0) or 0 for target in active_segments]
        threshold = max(thresholds) if thresholds else 0
        if threshold > 0 and len(existing_points) == len(active_segments):
            for idx, cached in enumerate(existing_points):
                ids = segment_ids_by_index.get(idx, [])
                if not ids:
                    continue
                mask_now = np.isin(segmentation_2d, ids)
                if not np.any(mask_now):
                    continue
                coords_now = np.argwhere(mask_now)
                center_now = (int(coords_now[:, 0].mean()), int(coords_now[:, 1].mean()))
                if max(abs(center_now[0] - int(cached[0])), abs(center_now[1] - int(cached[1]))) > threshold:
                    refresh_on_move = True
                    break

    if current_subgoal_segment != previous_subgoal_segment or refresh_on_move:

        def compute_center_from_ids(segmentation_mask: np.ndarray, ids: Iterable[int]):
            nonlocal no_object_flag
            ids = list(ids)
            if not ids:
                return None
            mask = np.isin(segmentation_mask, ids)
            if not np.any(mask):
                no_object_flag = True
                return None
            coords = np.argwhere(mask)
            if coords.size == 0:
                return None
            center_y = int(coords[:, 0].mean())
            center_x = int(coords[:, 1].mean())
            return [center_y, center_x]

        segment_centers: List[Optional[List[int]]] = []
        if active_segments:
            for idx in range(len(active_segments)):
                segment_centers.append(
                    compute_center_from_ids(
                        segmentation_2d, segment_ids_by_index.get(idx, [])
                    )
                )
        else:
            segment_centers.append(
                compute_center_from_ids(segmentation_2d, vis_obj_id_list)
            )

        segmentation_points = [center for center in segment_centers if center is not None]

        if current_subgoal_segment:
            normalized_centers: List[Optional[str]] = []
            for center in segment_centers:
                if center is None:
                    normalized_centers.append(None)
                    continue
                center_y, center_x = center
                normalized_centers.append(f"<{center_y}, {center_x}>")

            placeholder_pattern = re.compile(r"<[^>]*>")
            placeholders = list(placeholder_pattern.finditer(current_subgoal_segment))
            placeholder_count = len(placeholders)
            if placeholder_count > 0 and normalized_centers:
                replacements = normalized_centers.copy()
                if len(replacements) == 1 and placeholder_count > 1:
                    replacements = replacements * placeholder_count
                elif len(replacements) < placeholder_count:
                    replacements.extend([None] * (placeholder_count - len(replacements)))

                missing_placeholder = False
                new_text_parts: List[str] = []
                last_idx = 0
                for idx, match in enumerate(placeholders):
                    new_text_parts.append(
                        current_subgoal_segment[last_idx : match.start()]
                    )
                    replacement_text = replacements[idx]
                    if replacement_text is None:
                        missing_placeholder = True
                    else:
                        new_text_parts.append(replacement_text)
                    last_idx = match.end()
                new_text_parts.append(current_subgoal_segment[last_idx:])
                current_subgoal_segment_filled = (
                    current_task_name if missing_placeholder else "".join(new_text_parts)
                )
            else:
                current_subgoal_segment_filled = current_subgoal_segment
        else:
            current_subgoal_segment_filled = current_subgoal_segment

    return {
        "segmentation_result": segmentation_result,
        "segmentation_result_2d": segmentation_result_2d,
        "segmentation_points": segmentation_points,
        "current_subgoal_segment_filled": current_subgoal_segment_filled,
        "no_object_flag": no_object_flag,
        "updated_previous_subgoal_segment": current_subgoal_segment,
        "vis_obj_id_list": vis_obj_id_list,
    }


def create_segmentation_visuals(
    segmentation: np.ndarray,
    segmentation_result: np.ndarray,
    base_frame: np.ndarray,
    color_map: Dict[int, List[int]],
    segmentation_points: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build colored segmentation visualizations and target overlay for video export.

    Returns (segmentation_vis, segmentation_result_vis, target_for_video).
    """
    segmentation_for_video = copy.deepcopy(segmentation)
    segmentation_result_for_video = copy.deepcopy(segmentation_result)

    segmentation_vis = np.zeros(
        (*segmentation_for_video.shape[:2], 3), dtype=np.uint8
    )
    segmentation_result_vis = np.zeros(
        (*segmentation_result_for_video.shape[:2], 3), dtype=np.uint8
    )

    seg_2d = (
        segmentation_for_video.squeeze()
        if segmentation_for_video.ndim > 2
        else segmentation_for_video
    )
    seg_result_2d = (
        segmentation_result_for_video.squeeze()
        if segmentation_result_for_video.ndim > 2
        else segmentation_result_for_video
    )

    for seg_id in np.unique(seg_2d):
        if seg_id > 0:
            color = color_map.get(seg_id, [255, 255, 255])
            mask = seg_2d == seg_id
            segmentation_vis[mask] = color

    for seg_id in np.unique(seg_result_2d):
        if seg_id > 0:
            color = color_map.get(seg_id, [255, 255, 255])
            mask = seg_result_2d == seg_id
            segmentation_result_vis[mask] = color

    target_for_video = copy.deepcopy(base_frame)

    if segmentation_vis.shape[:2] != base_frame.shape[:2]:
        segmentation_vis = cv2.resize(
            segmentation_vis,
            (base_frame.shape[1], base_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if segmentation_result_vis.shape[:2] != base_frame.shape[:2]:
        segmentation_result_vis = cv2.resize(
            segmentation_result_vis,
            (base_frame.shape[1], base_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if segmentation_points:
        diameter = 5
        for center_y, center_x in segmentation_points:
            cv2.circle(target_for_video, (center_x, center_y), diameter, (255, 0, 0), -1)

    return segmentation_vis, segmentation_result_vis, target_for_video
