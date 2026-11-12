import os
import cv2
import numpy as np
from norfair import Detection, Tracker

path = os.path.expanduser("~/ball_tracker/data/soccernet/train/SNMOT-061/img1/%06d.jpg")
cap = cv2.VideoCapture(path, cv2.CAP_IMAGES)

if not cap.isOpened():
    raise RuntimeError(f"Could not closed sequence: {path}")

tracker = Tracker(
    distance_function="euclidean",  # plain pixel distance between points
    distance_threshold=40,          # px; match radius
    hit_counter_max=10,             # score cap (how long it survives unseen)
    initialization_delay=4,         # must be seen this many frames before it's reported
)

def get_that_pitch(hsv):
    H, W = hsv.shape[:2]
    downscaled = cv2.resize(hsv, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = downscaled.shape[:2]

    # hsv = cv2.cvtColor(downscaled, cv2.COLOR_BGR2HSV)

    lower = (35, 70, 40)
    upper = (85, 255, 255)
    thresholded = cv2.inRange(downscaled, lower, upper)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grande = max(contours, key=cv2.contourArea)
    johns_fav_hull = cv2.convexHull(grande)

    scale = np.array([W / w, H / h])
    hull_full = (johns_fav_hull * scale).astype(np.int32)

    pitch = np.zeros((H,W), np.uint8)
    cv2.drawContours(pitch, [hull_full], -1, 255, cv2.FILLED)

    return pitch

def get_those_ball_candidates(hsv, pitch):
    white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    green = cv2.inRange(hsv, (35, 70, 40), (85, 255, 255))
    # white = cv2.bitwise_and(white, pitch)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(white)
    ball_cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # if True:
        if 30 <= area <= 150:                           #Remove big blobs by area
            # if True:
            if max(w, h) <= 40:                         #No bb with either side bigger than 40px. Gets rid of long thin stuff pitchlines fx
                # if True:
                if 0.4 < w/h < 3.5:                     #bb is roughly square (motion blur accounted thus 2.5 instead of 1). Gets rid of arms, legs, etc
                    if area / (w*h) > 0.5:              #blob fills at least 40% of bb. circle should fill 78% of bb (pi/4)
                        cx, cy = centroids[i]                      # blob centre

                        R = int(max(w, h) * 1.2) + 4               # window half-size
                        x0, y0 = max(0, int(cx) - R), max(0, int(cy) - R)
                        x1, y1 = int(cx) + R + 1, int(cy) + R + 1
                        around = labels[y0:y1, x0:x1] == 0         # pixels that are NOT any white blob
                        green_ring = green[y0:y1, x0:x1][around].mean() / 255
                        # ball_cands.append(centroids[i])
                        if green_ring >= 0.9:
                            # ball_cands.append(centroids[i])
                            blob = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
                            cnt = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]

                            _, r = cv2.minEnclosingCircle(cnt)
                            # fill = area / (np.pi * r * r)
                            fill = area / (np.pi * (r + 0.5) ** 2)
                            if fill > 0.5:
                                ball_cands.append(centroids[i])

    return ball_cands, white

def range_score(v, low, high, soft):
    if low <= v <= high:
        return 1.0
    d = low - v if v < low else v - high
    return np.exp(-0.5 * (d / soft) ** 2)
    # return max(0.0, 1.0 - d / soft)       # linear: 0 at `soft` away

def score_those_ball_cands(hsv, pitch):
    white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    green = cv2.inRange(hsv, (35, 70, 40), (85, 255, 255))
    white = cv2.bitwise_and(white, pitch)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(white)
    ball_cands = []

    # Realistic (low, high) range for a ~10 px ball. Inside the range = ball-like.
    ideal_vals = {
        "area":        (30, 150, 60),    # px, data-dependent: check against gt.txt
        "max_side":    (9, 15, 8),      # px, max(w, h), data-dependent
        "aspect":      (1.0, 1.3, 1.2),   # max(w,h) / min(w,h). up to ~2.5 with motion blur
        "extent":      (0.75, 0.85, 0.2), # area / (w*h). perfect disc = pi/4 = 0.785
        "fill":        (0.6, 1.0, 0.3),   # area / (pi * (r + 0.5)**2)
        "green_ring":  (0.85, 1.0, 0.4),  # mean green around the blob
        "solidity":    (0.7, 1.0, 0.2),   # area / convex hull area
        "circularity": (0.6, 0.9, 0.4),   # 4*pi*A / P**2, unreliable at this size
        "elongation":  (0.8, 1.0, 0.5),   # sqrt(l2 / l1) from moments. ~0.4 with motion blur
    }

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        scores = []

        area_score = range_score(area, ideal_vals["area"][0], ideal_vals["area"][1], ideal_vals["area"][2])
        if area_score < 0.5:
            continue
        scores.append(area_score)

        max_side_score = range_score(max(w,h), ideal_vals["max_side"][0], ideal_vals["max_side"][1], ideal_vals["max_side"][2])
        if max_side_score < 0.5:
            continue
        scores.append(max_side_score)

        aspect_score = range_score(max(w, h) / min(w, h), ideal_vals["aspect"][0], ideal_vals["aspect"][1], ideal_vals["aspect"][2])
        if aspect_score < 0.5:
            continue
        scores.append(aspect_score)

        extent_score = range_score(area/(w*h), ideal_vals["extent"][0], ideal_vals["extent"][1], ideal_vals["extent"][2])
        if extent_score < 0.5:
            continue
        scores.append(extent_score)

        blob = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        cnt = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]

        _, r = cv2.minEnclosingCircle(cnt)
        # fill = area / (np.pi * r * r)
        fill = area / (np.pi * (r + 0.5) ** 2)

        fill_score = range_score(fill, ideal_vals["fill"][0], ideal_vals["fill"][1], ideal_vals["fill"][2])
        scores.append(fill_score)

        cx, cy = centroids[i]
        R = int(max(w, h) * 1.2) + 4               # window half-size
        x0, y0 = max(0, int(cx) - R), max(0, int(cy) - R)
        x1, y1 = int(cx) + R + 1, int(cy) + R + 1
        around = labels[y0:y1, x0:x1] == 0         # pixels that are NOT any white blob
        green_ring = green[y0:y1, x0:x1][around].mean() / 255

        green_ring_score = range_score(green_ring, ideal_vals["green_ring"][0], ideal_vals["green_ring"][1], ideal_vals["green_ring"][2])
        scores.append(green_ring_score)

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area > 0 else 0.0

        solidity_score = range_score(solidity, ideal_vals["solidity"][0], ideal_vals["solidity"][1], ideal_vals["solidity"][2])
        scores.append(solidity_score)

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else 0.0

        circularity_score = range_score(circularity, ideal_vals["circularity"][0], ideal_vals["circularity"][1], ideal_vals["circularity"][2])
        scores.append(circularity_score)

        m = cv2.moments(blob, binaryImage=True)
        mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
        common = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
        l1 = (mu20 + mu02 + common) / 2
        l2 = (mu20 + mu02 - common) / 2
        elongation = np.sqrt(l2 / l1) if l1 > 0 else 0.0

        elongation_score = range_score(elongation, ideal_vals["elongation"][0], ideal_vals["elongation"][1], ideal_vals["elongation"][2])
        scores.append(elongation_score)

        average = np.mean(scores)
        product = np.prod(scores)
        geo_mean = np.exp(np.mean(np.log(np.maximum(scores, 1e-6))))

        ball_cands.append({
            "label": i,                 # which component in `labels`
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "score": geo_mean,
            "scores": scores,           # per-measure breakdown
        })

    ball_cands.sort(key=lambda c: c["score"], reverse=True)    # best first
    return ball_cands, white





frame_amount = 0
cands_total = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    pitch = get_that_pitch(hsv)
    # ball_cands, white = get_those_ball_candidates(hsv, pitch)
 
    # detections = [Detection(points=np.array([[cx, cy]])) for cx, cy in ball_cands]
    # tracked = tracker.update(detections=detections)

    vis = frame.copy()

    # for obj in tracked:
    #     x, y = obj.estimate[0]          # filtered position
    #     cv2.circle(vis, (int(x), int(y)), 12, (0, 0, 255), 2)
    #     cv2.putText(vis, str(obj.id), (int(x) + 14, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    ball_cands, white = score_those_ball_cands(hsv, pitch)

    if ball_cands and ball_cands[0]["score"] > 0.6:
        ball = ball_cands[0]                         # best blob this frame
        bx, by = ball["centroid"]
        cv2.circle(vis, (int(bx), int(by)), 10, (0, 0, 255), 1)
        cv2.putText(vis,f'{ball["score"]:.2f}', (int(bx) + 12, int(by)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)

    # draw every candidate with its score
    # for c in ball_cands[:10]:                        # the top 10 is plenty
    #     cx, cy = c["centroid"]
    #     cv2.circle(vis, (int(cx), int(cy)), 10, (0, 0, 255), 1)
    #     cv2.putText(vis, f'{c["score"]:.2f}', (int(cx) + 12, int(cy)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)



    cv2.imshow("seq", vis)
    # print(len(ball_cands))
    cands_total += len(ball_cands)
    frame_amount += 1


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f'Average ball_cands pr frame {cands_total/frame_amount}')

cap.release()
cv2.destroyAllWindows()