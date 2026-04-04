"""
evaluate_generation.py
评估教师/学生或单模型扩散模型的生成质量，默认使用 FID + KID。

设计目标:
1. 比较教师与学生在统一评估分辨率下的生成分布质量
2. 兼容本项目的 DPM / DDPM 采样开关风格
3. 保存评估日志与结果，方便后续对比实验

依赖:
    pip install torch-fidelity

示例:
    python evaluate_generation.py ^
      --real_dir ../../autodl-fs/Houston2018/rgb ^
      --teacher_ckpt ../../autodl-tmp/kd_256/distil256_xxx/checkpoint-xxxxx ^
      --student_ckpt ../../autodl-tmp/kd_distill/distill_xxx/checkpoint-xxxxx ^
      --eval_size 128 ^
      --num_samples 5000 ^
      --use_dpm_solver ^
      --num_inference_steps 50

单模型评估（只输出该模型的 FID）示例:
    python evaluate_generation.py ^
      --real_dir ../../autodl-fs/Houston2018/rgb ^
      --model_ckpt ../../autodl-tmp/kd_distill/distil_xxx/checkpoint-xxxxx ^
      --model_size 128 ^
      --num_samples 5000 ^
      --use_dpm_solver ^
      --num_inference_steps 50
"""

import argparse
import json
import logging
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from diffusers import DDPMPipeline, DPMSolverMultistepScheduler
from torch_fidelity import calculate_metrics
from tqdm import tqdm


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
log = logging.getLogger(__name__)


def setup_logging(log_dir: str) -> None:
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "eval.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def list_images(root_dir: str) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    paths = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if not paths:
        raise FileNotFoundError(f"在 {root_dir} 中未找到图片")
    return sorted(paths)


import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# 假设这是你原有的 list_images 函数
# def list_images(real_dir: str): ...

def _process_image(args):
    """
    独立出来的单图处理函数。
    注意：多进程要求此函数必须在模块顶层定义，不能嵌套在 prepare_real_images 内部。
    """
    img_path, out_path, eval_size, index = args
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if eval_size > 0:
                img = img.resize((eval_size, eval_size), Image.BICUBIC)
            
            save_path = out_path / f"real_{index:06d}.png"
            img.save(save_path)
        return True
    except Exception as e:
        # 可选：打印错误以便调试，但不中断整体流程
        # print(f"Error processing {img_path}: {e}")
        return False

def prepare_real_images(real_dir: str, out_dir: str, eval_size: int) -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    real_paths = list_images(real_dir)
    
    # 准备参数列表：每个元素是一个元组 (图片路径, 输出目录, 尺寸, 索引)
    tasks = [(p, out_path, eval_size, i) for i, p in enumerate(real_paths)]
    
    # 确定进程数：默认使用所有可用 CPU 核心
    # 如果是在笔记本上运行且不想风扇狂转，可以设为 cpu_count() - 1
    workers = cpu_count()
    
    success_count = 0
    
    # 使用进程池并行处理
    # chunksize 优化大块数据传递效率，通常设为总任务数 / 进程数 / 4 左右
    chunksize = max(1, len(tasks) // (workers * 4))
    
    with Pool(processes=workers) as pool:
        # imap_unordered 比 map 更节省内存且能更快返回结果（不保证顺序，但我们需要的是进度条）
        # 使用 tqdm 包裹迭代器以显示进度
        for result in tqdm(
            pool.imap_unordered(_process_image, tasks, chunksize=chunksize), 
            total=len(real_paths), 
            desc="Preparing real images"
        ):
            if result:
                success_count += 1
                
    return success_count


def load_pipeline(ckpt: str, device: torch.device, use_dpm_solver: bool) -> DDPMPipeline:
    pipe = DDPMPipeline.from_pretrained(ckpt).to(device)
    pipe.set_progress_bar_config(disable=True)

    if use_dpm_solver:
        scheduler_dpm_path = Path(ckpt) / "scheduler_dpm"
        if scheduler_dpm_path.is_dir():
            pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(str(scheduler_dpm_path))
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe


@torch.no_grad()
def sample_images(
    ckpt: str,
    out_dir: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    eval_size: int,
    use_dpm_solver: bool,
    num_inference_steps: int,
    device: torch.device,
    name: str,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(ckpt, device, use_dpm_solver=use_dpm_solver)
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    total_batches = math.ceil(num_samples / batch_size)
    ddpm_steps = 1000
    actual_steps = num_inference_steps if use_dpm_solver else ddpm_steps

    log.info(
        f"[{name}] 开始采样 | checkpoint={ckpt} | "
        f"scheduler={'DPM' if use_dpm_solver else 'DDPM'} | steps={actual_steps}"
    )

    img_idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"Sampling {name}"):
        current_bs = min(batch_size, num_samples - img_idx)
        generator = torch.Generator(device=generator_device).manual_seed(seed + batch_idx)

        images = pipe(
            batch_size=current_bs,
            generator=generator,
            num_inference_steps=actual_steps,
        ).images

        for img in images:
            img = img.convert("RGB")
            if eval_size > 0:
                img = img.resize((eval_size, eval_size), Image.BICUBIC)
            img.save(out_path / f"sample_{img_idx:06d}.png")
            img_idx += 1


def evaluate_one_model(name: str, fake_dir: str, real_dir: str, use_cuda: bool) -> dict:
    log.info(f"[{name}] 开始计算 FID / KID")
    metrics = calculate_metrics(
        input1=fake_dir,
        input2=real_dir,
        cuda=use_cuda,
        fid=True,
        kid=True,
        isc=False,
        verbose=False,
    )

    result = {
        "fid": float(metrics["frechet_inception_distance"]),
        "kid_mean": float(metrics["kernel_inception_distance_mean"]),
        "kid_std": float(metrics["kernel_inception_distance_std"]),
    }

    log.info(f"[{name}] FID: {result['fid']:.4f}")
    log.info(f"[{name}] KID: {result['kid_mean']:.6f} ± {result['kid_std']:.6f}")
    return result


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="评估教师/学生或单模型扩散生成质量")

    parser.add_argument("--real_dir", type=str, required=True, help="真实图像目录")
    # 蒸馏评估模式：teacher_ckpt + student_ckpt
    parser.add_argument("--teacher_ckpt", type=str, default="", help="教师 checkpoint 路径（蒸馏模式）")
    parser.add_argument("--student_ckpt", type=str, default="", help="学生 checkpoint 路径（蒸馏模式）")
    # 单模型评估模式：model_ckpt
    parser.add_argument("--model_ckpt", type=str, default="", help="单模型 checkpoint 路径（单模型模式）")
    parser.add_argument("--model_size", type=int, default=0, help="单模型分辨率；当 --eval_size<=0 时用于推断 eval_size")

    parser.add_argument(
        "--eval_size",
        type=int,
        default=-1,
        help="评估时统一 resize 到该分辨率；若比较低分辨率语义能力，建议 128",
    )
    parser.add_argument("--num_samples", type=int, default=5000, help="每个模型生成多少张图用于评估")
    parser.add_argument("--batch_size", type=int, default=8, help="采样 batch size")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--use_dpm_solver", action="store_true", help="采样时使用 DPM 加速")
    parser.add_argument("--no_dpm_solver", dest="use_dpm_solver", action="store_false", help="采样时禁用 DPM，使用原始 DDPM")
    parser.set_defaults(use_dpm_solver=True)
    parser.add_argument("--num_inference_steps", type=int, default=50, help="仅 DPM 时生效的推理步数；DDPM 固定 1000")

    parser.add_argument("--output_dir", type=str, default="", help="评估结果输出目录；默认自动创建时间戳目录")
    parser.add_argument(
        "--keep_samples",
        action="store_true",
        help="保留中间生成图与真实图副本；默认只保留日志和 json 结果",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("eval_outputs", f"eval_{timestamp}")
    setup_logging(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # --------------------------
    # 入参校验与 eval 分辨率
    # --------------------------
    if args.model_ckpt:
        mode = "single"
        if args.teacher_ckpt or args.student_ckpt:
            log.info("[提醒] 检测到同时提供 teacher/student ckpt 与 model_ckpt，将以单模型模式为准。")
        if args.model_size <= 0:
            raise ValueError("--model_size 必须 > 0（当使用 --model_ckpt 时）")
        eval_size = args.eval_size if args.eval_size and args.eval_size > 0 else args.model_size
    else:
        mode = "distill"
        if not args.teacher_ckpt or not args.student_ckpt:
            raise ValueError("蒸馏模式需要同时提供 --teacher_ckpt 和 --student_ckpt；或提供 --model_ckpt 进入单模型模式。")
        eval_size = args.eval_size if args.eval_size and args.eval_size > 0 else 128

    log.info("========== 评估开始 ==========")
    log.info(f"[配置] real_dir={args.real_dir}")
    log.info(f"[配置] mode={mode}")
    log.info(f"[配置] teacher_ckpt={args.teacher_ckpt}")
    log.info(f"[配置] student_ckpt={args.student_ckpt}")
    log.info(f"[配置] model_ckpt={args.model_ckpt}")
    log.info(f"[配置] model_size={args.model_size}")
    log.info(f"[配置] eval_size={eval_size}")
    log.info(f"[配置] num_samples={args.num_samples}")
    log.info(f"[配置] batch_size={args.batch_size}")
    log.info(f"[配置] use_dpm_solver={args.use_dpm_solver}")
    log.info(f"[配置] num_inference_steps={args.num_inference_steps if args.use_dpm_solver else 1000}")
    log.info(f"[配置] device={device}")
    log.info(f"[配置] output_dir={output_dir}")

    temp_ctx = tempfile.TemporaryDirectory() if not args.keep_samples else None
    if args.keep_samples:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(temp_ctx.name)

    real_eval_dir = base_dir / "real_eval"
    teacher_eval_dir = base_dir / "teacher_eval"
    student_eval_dir = base_dir / "student_eval"

    real_count = prepare_real_images(args.real_dir, str(real_eval_dir), eval_size)
    log.info(f"[真实图] 已准备 {real_count} 张")

    if mode == "single":
        model_eval_dir = base_dir / "model_eval"
        model_name = "Model"

        sample_images(
            ckpt=args.model_ckpt,
            out_dir=str(model_eval_dir),
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            eval_size=eval_size,
            use_dpm_solver=args.use_dpm_solver,
            num_inference_steps=args.num_inference_steps,
            device=device,
            name=model_name,
        )

        model_scores = evaluate_one_model(model_name, str(model_eval_dir), str(real_eval_dir), use_cuda)

        summary = {
            "config": {
                "real_dir": args.real_dir,
                "model_ckpt": args.model_ckpt,
                "model_size": args.model_size,
                "eval_size": eval_size,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "use_dpm_solver": args.use_dpm_solver,
                "num_inference_steps": args.num_inference_steps if args.use_dpm_solver else 1000,
                "device": str(device),
            },
            "model": model_scores,
        }

        result_json = os.path.join(output_dir, "eval_result.json")
        save_json(summary, result_json)

        log.info("========== 评估完成 ==========")
        log.info(f"Model FID: {model_scores['fid']:.4f}")
        log.info(f"Model KID: {model_scores['kid_mean']:.6f} ± {model_scores['kid_std']:.6f}")
        log.info(f"结果已保存到: {result_json}")
    else:
        sample_images(
            ckpt=args.teacher_ckpt,
            out_dir=str(teacher_eval_dir),
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            eval_size=eval_size,
            use_dpm_solver=args.use_dpm_solver,
            num_inference_steps=args.num_inference_steps,
            device=device,
            name="Teacher",
        )

        sample_images(
            ckpt=args.student_ckpt,
            out_dir=str(student_eval_dir),
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed + 10000,
            eval_size=eval_size,
            use_dpm_solver=args.use_dpm_solver,
            num_inference_steps=args.num_inference_steps,
            device=device,
            name="Student",
        )

        teacher_scores = evaluate_one_model(
            "Teacher", str(teacher_eval_dir), str(real_eval_dir), use_cuda
        )
        student_scores = evaluate_one_model(
            "Student", str(student_eval_dir), str(real_eval_dir), use_cuda
        )

        summary = {
            "config": {
                "real_dir": args.real_dir,
                "teacher_ckpt": args.teacher_ckpt,
                "student_ckpt": args.student_ckpt,
                "eval_size": eval_size,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "use_dpm_solver": args.use_dpm_solver,
                "num_inference_steps": args.num_inference_steps if args.use_dpm_solver else 1000,
                "device": str(device),
            },
            "teacher": teacher_scores,
            "student": student_scores,
        }

        result_json = os.path.join(output_dir, "eval_result.json")
        save_json(summary, result_json)

        log.info("========== 评估完成 ==========")
        log.info(f"Teacher FID: {teacher_scores['fid']:.4f}")
        log.info(f"Student FID: {student_scores['fid']:.4f}")
        log.info(f"Teacher KID: {teacher_scores['kid_mean']:.6f} ± {teacher_scores['kid_std']:.6f}")
        log.info(f"Student KID: {student_scores['kid_mean']:.6f} ± {student_scores['kid_std']:.6f}")
        log.info(f"结果已保存到: {result_json}")

    if temp_ctx is not None:
        temp_ctx.cleanup()


if __name__ == "__main__":
    main()
