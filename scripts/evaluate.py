"""학습된 best.pt를 검증(val)·테스트(test) 세트에서 평가하고 부류별 결과 출력"""
import sys, json
from ultralytics import YOLO
model = YOLO(sys.argv[1]); out = {}
for split in ['val', 'test']:
    m = model.val(data='data.yaml', split=split, imgsz=416, batch=16, device='cpu',
                  plots=True, project='runs/eval', name=split, exist_ok=True, verbose=False)
    out[split] = {'all': dict(P=m.box.mp, R=m.box.mr, mAP50=m.box.map50, mAP50_95=m.box.map)}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        out[split][model.names[int(c)]] = dict(P=p, R=r, mAP50=ap50, mAP50_95=ap)
    out[split]['speed_ms'] = m.speed
print(json.dumps(out, indent=1, default=float))
json.dump(out, open('metrics.json', 'w'), indent=1, default=float)
