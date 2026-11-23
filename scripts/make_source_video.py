"""테스트 이미지(학습에 사용하지 않은 이미지)에 팬/줌(카메라 이동) 효과를 주어 5초 원본 영상 합성
사용: python make_source_video.py out.mp4 img1 img2 img3 ...
※ 실제 촬영 영상이 있다면 이 단계 없이 track_video.py에 바로 넣으면 됨"""
import sys, cv2, numpy as np
out, imgs = sys.argv[1], sys.argv[2:]
W, H, FPS, SEC = 960, 540, 30, 5.0
per = int(FPS * SEC / len(imgs))
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
for k, p in enumerate(imgs):
    im = cv2.imread(p); h, w = im.shape[:2]
    s = max(W / w, H / h) * 1.35                    # 화면보다 크게 확대해 이동 여유 확보
    im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC); h, w = im.shape[:2]
    for i in range(per):
        t = i / (per - 1)
        z = 1.0 + 0.25 * (t if k % 2 == 0 else 1 - t)   # 줌 인/아웃 교대
        cw, ch = int(W * z), int(H * z)
        cw, ch = min(cw, w), min(ch, h)
        dx = (t if k % 2 == 0 else 1 - t)                # 좌↔우 팬
        x0 = int((w - cw) * dx); y0 = int((h - ch) * 0.5)
        frame = cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
        vw.write(frame)
vw.release(); print('saved', out, per * len(imgs), 'frames')
