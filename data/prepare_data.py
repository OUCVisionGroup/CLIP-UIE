import argparse
from io import BytesIO
from multiprocessing import Lock, Process, RawValue
from functools import partial
from PIL import Image
import os
from pathlib import Path
import lmdb
import numpy as np
import time

def resize_and_convert(img, size, resample):
    if img.size[0] != size:
        img = img.resize((size, size), resample)
        img = img.crop((0, 0, size, size))
    return img

def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()

def resize_multiple(img, sizes=(16, 128), resample=Image.Resampling.BICUBIC, lmdb_save=False):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(img, sizes[1], resample)
    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)
    return [lr_img, hr_img, sr_img]

def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file).convert('RGB')
    return img_file.name.split('.')[0], resize_multiple(img, sizes, resample, lmdb_save)

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes
        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, (lr_img, _, sr_img) = wctx.resize_fn(file)
        if not wctx.lmdb_save:
            lr_img.save(f'{wctx.out_path}/lr_{wctx.sizes[0]}/{i.zfill(5)}.png')
            sr_img.save(f'{wctx.out_path}/sr_{wctx.sizes[0]}_{wctx.sizes[1]}/{i.zfill(5)}.png')
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put(f'lr_{wctx.sizes[0]}_{i.zfill(5)}'.encode(), lr_img)
                txn.put(f'sr_{wctx.sizes[0]}_{wctx.sizes[1]}_{i.zfill(5)}'.encode(), sr_img)
                txn.put('length'.encode(), str(wctx.inc_get()).encode())
        wctx.inc_get()

def prepare_process_worker_ref(wctx, file_subset):
    for file in file_subset:
        i, (_, hr_img, _) = wctx.resize_fn(file)
        if not wctx.lmdb_save:
            hr_img.save(f'{wctx.out_path}/hr_{wctx.sizes[1]}/{i.zfill(5)}.png')
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put(f'hr_{wctx.sizes[1]}_{i.zfill(5)}'.encode(), hr_img)
                txn.put('length'.encode(), str(wctx.inc_get()).encode())
        wctx.inc_get()

def all_threads_inactive(worker_threads):
    return all(not thread.is_alive() for thread in worker_threads)

# 省略前面代码，直接修改 prepare 函数相关部分

def prepare(raw_img_path, ref_img_path, out_path, n_worker, sizes, resample, lmdb_save):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample, lmdb_save=lmdb_save)

    raw_files_set = {str(p.relative_to(raw_img_path)) for p in Path(raw_img_path).glob('**/*')}
    ref_files_set = {str(p.relative_to(ref_img_path)) for p in Path(ref_img_path).glob('**/*')}

    if raw_files_set != ref_files_set:
        print("❌ Directory structures do not match. Aborting.")
        return

    relative_paths = sorted(raw_files_set)
    matched_raw_files = [Path(raw_img_path) / p for p in relative_paths]
    matched_ref_files = [Path(ref_img_path) / p for p in relative_paths]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(f'{out_path}/lr_{sizes[0]}', exist_ok=True)
        os.makedirs(f'{out_path}/hr_{sizes[1]}', exist_ok=True)
        os.makedirs(f'{out_path}/sr_{sizes[0]}_{sizes[1]}', exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    if n_worker > 1:
        multi_env = env if lmdb_save else None
        # 分别创建 raw 和 ref 的计数上下文
        wctx_raw = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)
        wctx_ref = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        file_subsets_raw = np.array_split(matched_raw_files, n_worker)
        file_subsets_ref = np.array_split(matched_ref_files, n_worker)
        worker_threads = []

        # 启动 raw 图像处理线程
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx_raw, file_subsets_raw[i]))
            proc.start()
            worker_threads.append(proc)

        # 启动 ref 图像处理线程
        for i in range(n_worker):
            proc_ref = Process(target=prepare_process_worker_ref, args=(wctx_ref, file_subsets_ref[i]))
            proc_ref.start()
            worker_threads.append(proc_ref)

        # 分别显示 raw 和 ref 的处理进度
        while not all_threads_inactive(worker_threads):
            print(f"\rRaw: {wctx_raw.value()}/{len(matched_raw_files)} | Ref: {wctx_ref.value()}/{len(matched_ref_files)} images processed", end=" ")
            time.sleep(0.1)
        print()  # 换行
    else:
        # 单线程处理也分开计数，分别保存
        for file in matched_raw_files:
            i, (lr_img, _, sr_img) = resize_fn(file)
            lr_img.save(f'{out_path}/lr_{sizes[0]}/{i.zfill(5)}.png')
            sr_img.save(f'{out_path}/sr_{sizes[0]}_{sizes[1]}/{i.zfill(5)}.png')
        for file in matched_ref_files:
            i, (_, hr_img, _) = resize_fn(file)
            hr_img.save(f'{out_path}/hr_{sizes[1]}/{i.zfill(5)}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_raw', '-p1', type=str, default='/data/liusx/Pycharm/CLIP-UIE/dataset/demo/raw')
    parser.add_argument('--path_ref', '-p2', type=str, default='/data/liusx/Pycharm/CLIP-UIE/dataset/demo/ref')
    parser.add_argument('--out', '-o', type=str, default='./dataset/test_demo')
    parser.add_argument('--size', type=str, default='32, 256')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC
    }
    sizes = [int(s.strip()) for s in args.size.split(',')]
    args.out = f'{args.out}_{sizes[0]}_{sizes[1]}'
    prepare(args.path_raw, args.path_ref, args.out, args.n_worker, sizes, resample_map[args.resample], args.lmdb)
