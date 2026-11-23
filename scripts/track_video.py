"""fine-tuning된 모델로 동영상에서 객체 감지 + 추적(ByteTrack), 궤적(trajectory) 표시
사용: python track_video.py best.pt input.mp4 output.mp4"""
import sys, cv2, collections, numpy as np
from ultralytics import YOLO
wts, src, out = sys.argv[1:4]
model = YOLO(wts)
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
track_hist = collections.defaultdict(list)          # 객체 ID별 중심점 궤적 저장
ids_seen = collections.defaultdict(set)
prev = None
while True:
    ok, frame = cap.read()
    if not ok: break
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
    if prev is not None and cv2.absdiff(small, prev).mean() > 40:   # 장면 전환 감지 → 추적기·궤적 초기화
        track_hist.clear()
        if getattr(model, 'predictor', None) and getattr(model.predictor, 'trackers', None):
            for t in model.predictor.trackers: t.reset()
    prev = small
    r = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.35, imgsz=416, verbose=False)[0]
    vis = r.plot(line_width=2)
    if r.boxes.id is not None:
        for (x, y, w, h), tid, c in zip(r.boxes.xywh.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
            track_hist[tid].append((int(x), int(y))); track_hist[tid] = track_hist[tid][-40:]
            ids_seen[model.names[c]].add(tid)
            pts = np.array(track_hist[tid], np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (230, 230, 230), 3)
    cv2.rectangle(vis, (0, H - 34), (W, H), (0, 0, 0), -1)               # 하단 캡션 띠
    cv2.putText(vis, 'YOLO11n fine-tuned: elephant / rhino / zebra  |  ByteTrack tracking', (12, H - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(vis)
cap.release(); vw.release()
print('saved', out, {k: len(v) for k, v in ids_seen.items()})
