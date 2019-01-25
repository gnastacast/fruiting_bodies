"""
MIT License

Copyright (c) 2017 TU Berlin, Communication Systems Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import time

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

class IouTracker(object):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
         sigma_l (float): low detection threshold.
         sigma_iou (float): IOU threshold.
         t_max (float): number of seconds until frame is deleted.
         smooth_window (int) : number of detections over which to average score

    Returns:
        list: list of tracks.
    """
    def __init__(self, sigma_l = 0.0, sigma_iou = 0.3, t_max = 0.25, smooth_window = 10):
        # Initialize variables
        self.sigma_l = sigma_l
        self.sigma_iou = sigma_iou
        self.t_max = t_max
        #Initialize tracking
        self._tracks_active = []
        self._next_id = 0
        self.smooth_window = smooth_window

    def update(self, datum, update_time=None):
        '''
        Args:
             datum (openpose data structure)
        '''
        dets = self._parse_datum(datum)
        dets = [det for det in dets if det['score'] >= self.sigma_l]
        pose_ids = [i for i in range(len(dets))]
        updated_tracks = []
        if update_time is None:
            update_time = time.time()
        for track in self._tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bbox'], x['bbox']))
                if iou(track['bbox'], best_match['bbox']) >= self.sigma_iou:
                    track['bbox'] = best_match['bbox']
                    track['score'] = ((self.smooth_window - 1)*track['score'] + best_match['score']) / self.smooth_window
                    track['last_seen'] = update_time
                    updated_tracks.append(track)
                    best_idx = dets.index(best_match)
                    track['pose_id'] = pose_ids[best_idx]
                    # remove from best matching detection from detections
                    del dets[best_idx]
                    del pose_ids[best_idx]

             # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if update_time - track['last_seen']  < self.t_max:
                    track['score'] = self._lerp_score(track['score'], 0)
                    track['pose_id'] = -1
                    updated_tracks.append(track)

        # create new tracks
        new_ids = np.arange(len(dets)) + self._next_id
        new_tracks = [{'id'          : i,
                       'pose_id'     : idx,
                       'bbox'        : det['bbox'],
                       'score'       : det['score'],
                       'last_seen'   : update_time} for i, idx, det in zip(new_ids, pose_ids, dets)]

        self._next_id += len(new_ids)        
        self._tracks_active = updated_tracks + new_tracks

    def _parse_datum(self, datums, ids=None):
        ret = []
        if len(datums.poseKeypoints.shape):
            for points, score in zip(datums.poseKeypoints, datums.poseScores):
                min_x, min_y, _ = np.min(points[points[:,2] > 0.01], axis=0)
                max_x, max_y, _ = np.max(points[points[:,2] > 0.01], axis=0)
                if ids is not None:
                    pass
                ret.append({'bbox': [min_x, min_y, max_x, max_y], 'score': score})
        
        return ret

    def _lerp_score(self, old, new):
        return ((self.smooth_window - 1)*old + new) / self.smooth_window

    def get_tracks(self):
        return [{'score'  : tr['score'],
                 'center' : np.mean(np.array(tr['bbox']).reshape(2,2), axis=0),
                 'pose_id': tr['pose_id'],
                 'id'     : tr['id']} for tr in self._tracks_active]