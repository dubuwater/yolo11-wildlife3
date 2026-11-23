"""african-wildlife(4부류)에서 elephant, rhino, zebra 3개 부류만 추출해 YOLO 데이터셋 구성"""
import glob, os, shutil, random, sys, collections
src, dst = sys.argv[1], sys.argv[2]
KEEP = {1: 0, 2: 1, 3: 2}            # elephant->0, rhino->1, zebra->2 (buffalo=0 제외)
NAMES = ['elephant', 'rhino', 'zebra']
MAX_TRAIN_PER_CLASS = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
random.seed(0)
for split in ['train', 'val', 'test']:
    os.makedirs(f'{dst}/images/{split}', exist_ok=True); os.makedirs(f'{dst}/labels/{split}', exist_ok=True)
    files = sorted(glob.glob(f'{src}/labels/{split}/*.txt')); random.shuffle(files)
    per = collections.Counter(); n = 0
    for lf in files:
        rows = [l.split() for l in open(lf) if l.strip()]
        cls = {int(r[0]) for r in rows}
        if not rows or 0 in cls: continue           # buffalo 포함 이미지 제외
        main = min(cls, key=lambda c: -sum(int(r[0]) == c for r in rows))
        if split == 'train' and per[main] >= MAX_TRAIN_PER_CLASS: continue
        per[main] += 1; n += 1
        stem = os.path.splitext(os.path.basename(lf))[0]
        img = glob.glob(f'{src}/images/{split}/{stem}.*')[0]
        shutil.copy(img, f'{dst}/images/{split}/')
        with open(f'{dst}/labels/{split}/{stem}.txt', 'w') as f:
            f.writelines(' '.join([str(KEEP[int(r[0])])] + r[1:]) + '\n' for r in rows)
    print(split, n, {NAMES[KEEP[k]]: v for k, v in sorted(per.items())})
with open(f'{dst}/data.yaml', 'w') as f:
    f.write(f"path: {os.path.abspath(dst)}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n" + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES)))
