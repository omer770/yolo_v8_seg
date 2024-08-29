import json
import copy
import numpy as np
import pandas as pd
from shapely import geometry
from typing import List, Dict, Tuple, Any
from shapely.geometry import Polygon

def is_valid_segmentation(segmentation)->bool:
  """
  Checks if a segmentation mask (represented as a list of coordinates) is valid using shapely.

  Args:
      segmentation: A list of coordinates representing the segmentation mask.

  Returns:
      True if the segmentation is valid, False otherwise.
  """
  try:
    poly = Polygon(segmentation)
    return poly.is_valid
  except:
    return False

def generate_poly_from_list(segmentation_list: str | list) -> Polygon:
  """Generates a Shapely Polygon object from a list of coordinates.

  Args:
      segmentation_list: A list of coordinates or a string representation of a list of coordinates.

  Returns:
      A Shapely Polygon object.
  """

  if isinstance(segmentation_list, str):
    segmentation_list = segmentation_list.replace('[', '').replace(']', '').split(',')
    segmentation_list = [[int(segmentation_list[p]), int(segmentation_list[p + 1])] for p in range(0, len(segmentation_list), 2)]

  if segmentation_list[0] != segmentation_list[-1]:
    if abs(segmentation_list[0][0] - segmentation_list[-1][0]) <= 7 and abs(segmentation_list[0][1] - segmentation_list[-1][1]) <= 7:
      segmentation_list[-1] = segmentation_list[0]
    else:
      segmentation_list.append(segmentation_list[0])

  return Polygon(segmentation_list)

def calculate_iou(seg1: List[float], seg2: List[float]) -> float:
    """Calculates the Intersection over Union (IoU) between two segmentation masks.

    Args:
        seg1: The first segmentation mask as a list of coordinates.
        seg2: The second segmentation mask as a list of coordinates.

    Returns:
        The IoU value between the two masks.
    """
    poly1 = generate_poly_from_list(seg1)
    poly2 = generate_poly_from_list(seg2)
    return poly1.intersection(poly2).area / poly1.union(poly2).area

def non_max_sup(pred_dict, conf_threshold, nms_threshold, verbose=False,return_dropped_objects:bool= False):
  """
  Performs Non-Maximum Suppression (NMS) on detections, removing those with invalid geometries.

  Args:
      pred_object: A dictionary where keys are file names and values are dictionaries containing detections for that file.
          Each detection dictionary should have 'confidence' (float), 'classname' (str), and 'segmentation' (data structure representing segmentation) keys.
      threshold: Minimum confidence score to keep a detection.
      nms_threshold: Intersection-over-Union (IOU) threshold for suppression.
      verbose: (Optional) Boolean flag to enable verbose output. Defaults to False.

  Returns:
      A dictionary where keys are file names and values are lists of names of detections that were removed due to NMS or invalid geometry.
  """
  pred_object = copy.deepcopy(pred_dict)
  #dout = {k:v for k,v in pred_dict.items()}# Make a copy to avoid modifying the original dictionary
  remove_dict = {}
  for fname, detections in pred_object.items():
    # Filter detections with invalid geometries
    detections = {k: v for k, v in detections.items()}

    # Sort detections by confidence (descending)
    sorted_detections = sorted(detections.items(), key=lambda x: x[1]['confidence'], reverse=True)

    # Keep only detections above the threshold
    detections_to_keep = [detection for detection in sorted_detections if detection[1]['confidence'] >= conf_threshold]
    removed_detections = [detection[0] for detection in sorted_detections if detection not in detections_to_keep]

    # Perform NMS on remaining detections
    for i, det_i in enumerate(detections_to_keep):
      for j, det_j in enumerate(detections_to_keep[i + 1:]):
        try: 
          iou_nms = calculate_iou(det_i[1]['segmentation'], det_j[1]['segmentation'])
          if iou_nms >= nms_threshold:
            # Remove detection with lower confidence
            removed_detections.append(det_j[0])
          try:
            del pred_object[fname][det_j[0]]  # Remove detection from original dictionary
          except KeyError:
            pass  # Ignore if key already removed
          if verbose:
            print(f"\tFile - {fname}: Removed detection '{det_j[0]}' (iou: {iou_nms:.4f})")
        except: removed_detections.append(det_j[0])

    # Update remove dictionary with names of removed detections
    remove_dict[fname] = removed_detections
    removed_dict = {k: v for k, v in remove_dict.items() if v}
  if return_dropped_objects:
    return pred_object,removed_dict
  return pred_object
