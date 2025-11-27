__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    
    OPTIMIZED VERSION - Modified to handle large datasets with limited memory
    Modified by: IA Assistant
    Date: November 23, 2025
    """
__author__ = "Alexandre Delplanque (optimized by IA Assistant)"
__license__ = "MIT License"
__version__ = "0.2.1-optimized"


import argparse
import torch
import os
import pandas
import warnings
import numpy
import PIL
import gc
import psutil

import albumentations as A

from torch.utils.data import DataLoader, Subset
from PIL import Image
from pathlib import Path

from animaloc.data.transforms import DownSample, Rotate90
from animaloc.models import LossWrapper, HerdNet
from animaloc.eval import HerdNetStitcher, HerdNetEvaluator
from animaloc.eval.metrics import PointsMetrics
from animaloc.datasets import CSVDataset
from animaloc.utils.useful_funcs import mkdir, current_date
from animaloc.vizual import draw_points, draw_text

warnings.filterwarnings('ignore')
PIL.Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser(
    prog='inference_optimized', 
    description='Collects the detections of a pretrained HerdNet model on a set of images '
                'with optimized memory usage (processes images in batches)'
    )

parser.add_argument('root', type=str,
    help='path to the JPG images folder (str)')
parser.add_argument('pth', type=str,
    help='path to PTH file containing your model parameters (str)')  
parser.add_argument('-size', type=int, default=512,
    help='patch size use for stitching. Defaults to 512.')
parser.add_argument('-over', type=int, default=160,
    help='overlap for stitching. Defaults to 160.')
parser.add_argument('-device', type=str, default='cuda',
    help='device on which model and images will be allocated (str). \
        Possible values are \'cpu\' or \'cuda\'. Defaults to \'cuda\'.')
parser.add_argument('-ts', type=int, default=256,
    help='thumbnail size. Defaults to 256.')
parser.add_argument('-pf', type=int, default=10,
    help='print frequence. Defaults to 10.')
parser.add_argument('-rot', type=int, default=0,
    help='number of times to rotate by 90 degrees. Defaults to 0.')
parser.add_argument('-batch', type=int, default=5,
    help='number of images to process per batch (memory optimization). Defaults to 5.')
parser.add_argument('-batch_patches', type=int, default=8,
    help='number of patches to process per batch in stitcher (GPU). For 16GB GPU, 8-12 is optimal. Defaults to 8.')
parser.add_argument('-workers', type=int, default=4,
    help='number of worker threads for data loading and visualization export. Defaults to 4.')

args = parser.parse_args()


# ============================================================
# MEMORY MONITORING FUNCTIONS
# ============================================================
def print_memory_usage(stage=""):
    """Print current RAM and GPU memory usage"""
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3
    
    gpu_mem = ""
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_max_gb = torch.cuda.max_memory_allocated() / 1024**3
        gpu_mem = f"| GPU: {gpu_gb:.2f}GB (max: {gpu_max_gb:.2f}GB)"
    
    print(f"[MEMORY {stage}] RAM: {ram_gb:.2f}GB {gpu_mem}")


def clear_memory():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# OPTIMIZED STITCHER
# ============================================================
class OptimizedHerdNetStitcher(HerdNetStitcher):
    """
    Memory-optimized stitcher that processes patches in small batches
    """
    
    def __init__(self, *args, batch_size_patches=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size_patches
    
    @torch.no_grad()
    def _inference(self, patches):
        """
        GPU-optimized inference with larger batches and minimal CPU transfers.
        """
        self.model.eval()
        
        import torch.nn.functional as F
        from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
        
        # Calculate optimal batch size based on available GPU memory
        # For 16GB GPU, we can safely process 8-12 patches at once (512x512)
        # Each patch ~50-100MB, so 8 patches = ~800MB + model ~2GB = ~3GB total
        optimal_batch_size = self.batch_size if self.batch_size > 1 else 8
        
        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset,
            batch_size=optimal_batch_size,  # Process multiple patches at once
            sampler=SequentialSampler(dataset),
            pin_memory=True  # Faster CPU→GPU transfer
        )
        
        maps = []
        for batch_idx, patch in enumerate(dataloader):
            patch = patch[0].to(self.device, non_blocking=True)  # Async transfer
            outputs, _ = self.model(patch)
            
            # Keep everything on GPU as long as possible
            if isinstance(outputs, tuple):
                # HerdNet returns (heatmap, clsmap) with different spatial sizes
                # Do ALL processing on GPU before moving to CPU
                heatmap_gpu = outputs[0]  # [B, 1, H, W] on GPU
                clsmap_gpu = outputs[1]   # [B, num_classes, h, w] on GPU
                
                # Upsample clsmap on GPU (much faster than CPU)
                if clsmap_gpu.shape[2:] != heatmap_gpu.shape[2:]:
                    clsmap_gpu = F.interpolate(
                        clsmap_gpu, 
                        size=heatmap_gpu.shape[2:],
                        mode='bilinear', 
                        align_corners=True
                    )
                
                # Concatenate on GPU
                outputs_gpu = torch.cat([heatmap_gpu, clsmap_gpu], dim=1)
                
                # Only NOW move to CPU (single transfer instead of multiple)
                outputs_cpu = outputs_gpu.cpu()
            else:
                outputs_cpu = outputs.cpu()
            
            # Split batch into individual tensors
            for i in range(outputs_cpu.shape[0]):
                maps.append(outputs_cpu[i:i+1])
            
            # Clear GPU memory every few batches
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return maps


def main():
    
    print("="*80)
    print("HERDNET INFERENCE - OPTIMIZED FOR MEMORY")
    print("="*80)
    print(f"Processing images from: {args.root}")
    print(f"Model: {args.pth}")
    print(f"Batch size (images): {args.batch}")
    print(f"Batch size (patches): {args.batch_patches}")
    print(f"Parallel workers: {args.workers}")
    print(f"Device: {args.device}")
    print("="*80)
    
    print_memory_usage("START")

    # Create destination folder
    curr_date = current_date()
    dest = os.path.join(args.root, f"{curr_date}_HerdNet_results")
    mkdir(dest)
    
    # Read info from PTH file
    map_location = torch.device('cpu')
    if torch.cuda.is_available():
        map_location = torch.device('cuda')

    print('\nLoading model...')
    checkpoint = torch.load(args.pth, map_location=map_location)
    classes = checkpoint['classes']
    num_classes = len(classes) + 1
    img_mean = checkpoint['mean']
    img_std = checkpoint['std']
    
    # Get list of images
    img_names = [i for i in os.listdir(args.root) 
            if i.endswith(('.JPG','.jpg','.JPEG','.jpeg'))]
    n = len(img_names)
    
    print(f'\nTotal images to process: {n}')
    print(f'Processing in batches of: {args.batch} images')
    
    if n == 0:
        print("ERROR: No images found!")
        return
    
    # Build the trained model
    print('\nBuilding the model ...')
    device = torch.device(args.device)
    model = HerdNet(num_classes=num_classes, pretrained=False)
    model = LossWrapper(model, [])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print_memory_usage("AFTER MODEL LOAD")
    
    # Build the optimized stitcher
    stitcher = OptimizedHerdNetStitcher(
            model=model,
            size=(args.size, args.size),
            overlap=args.over,
            batch_size_patches=args.batch_patches,
            down_ratio=2,
            up=True, 
            reduction='mean',
            device_name=device
            )

    # Prepare destination folders
    dest_plots = os.path.join(dest, 'plots')
    mkdir(dest_plots)
    dest_thumb = os.path.join(dest, 'thumbnails')
    mkdir(dest_thumb)
    
    # Prepare CSV for incremental saves
    detections_path = os.path.join(dest, f'{curr_date}_detections.csv')
    all_detections = []
    
    # Process images in batches
    print('\nStarting inference...')
    print("="*80)
    
    for batch_start in range(0, n, args.batch):
        batch_end = min(batch_start + args.batch, n)
        batch_img_names = img_names[batch_start:batch_end]
        batch_size = len(batch_img_names)
        
        print(f'\n[BATCH {batch_start//args.batch + 1}] Processing images {batch_start+1}-{batch_end} of {n}')
        print_memory_usage("BATCH START")
        
        # Create dataframe for this batch
        df_batch = pandas.DataFrame(data={
            'images': batch_img_names, 
            'x': [0]*batch_size, 
            'y': [0]*batch_size, 
            'labels': [1]*batch_size
        })
        
        # Prepare transforms
        end_transforms = []
        if args.rot != 0:
            end_transforms.append(Rotate90(k=args.rot))
        end_transforms.append(DownSample(down_ratio=2, anno_type='point'))
        
        albu_transforms = [A.Normalize(mean=img_mean, std=img_std)]
        
        # Create dataset for this batch
        dataset_batch = CSVDataset(
            csv_file=df_batch,
            root_dir=args.root,
            albu_transforms=albu_transforms,
            end_transforms=end_transforms
        )
        
        dataloader_batch = DataLoader(
            dataset_batch, 
            batch_size=1, 
            shuffle=False,
            num_workers=min(args.workers, batch_size),  # Use multiple workers for data loading
            sampler=torch.utils.data.SequentialSampler(dataset_batch),
            pin_memory=True if device == torch.device('cuda') else False  # Speed up GPU transfer
        )
        
        # Create metrics for this batch
        metrics_batch = PointsMetrics(5, num_classes=num_classes)
        
        # Create evaluator for this batch
        evaluator_batch = HerdNetEvaluator(
            model=model,
            dataloader=dataloader_batch,
            metrics=metrics_batch,
            lmds_kwargs=dict(kernel_size=(3,3), adapt_ts=0.2, neg_ts=0.1),
            device_name=device,
            print_freq=max(1, batch_size // 2),
            stitcher=stitcher,
            work_dir=dest,
            header=f'[BATCH {batch_start//args.batch + 1}]'
        )
        
        # Run inference on this batch
        try:
            evaluator_batch.evaluate(wandb_flag=False, viz=False, log_meters=True)
            
            # Get detections for this batch
            detections_batch = evaluator_batch.detections
            detections_batch.dropna(inplace=True)
            detections_batch['species'] = detections_batch['labels'].map(classes)
            
            # Append to all detections
            all_detections.append(detections_batch)
            
            # Save incrementally (append mode)
            if batch_start == 0:
                # First batch - create new file with header
                detections_batch.to_csv(detections_path, index=False, mode='w')
            else:
                # Subsequent batches - append without header
                detections_batch.to_csv(detections_path, index=False, mode='a', header=False)
            
            print(f'[BATCH {batch_start//args.batch + 1}] Detections saved incrementally: {len(detections_batch)} objects')
            
            # Process visualizations for this batch in parallel
            print(f'[BATCH {batch_start//args.batch + 1}] Exporting plots and thumbnails...')
            
            # Use ThreadPoolExecutor for I/O bound operations (image loading/saving)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_single_image(img_name):
                """Process visualization for a single image"""
                try:
                    if img_name not in detections_batch['images'].values:
                        return 0
                        
                    img = Image.open(os.path.join(args.root, img_name))
                    if args.rot != 0:
                        rot = args.rot * 90
                        img = img.rotate(rot, expand=True)
                    img_cpy = img.copy()
                    
                    img_dets = detections_batch[detections_batch['images']==img_name]
                    pts = list(img_dets[['y','x']].to_records(index=False))
                    pts = [(y, x) for y, x in pts]
                    
                    n_dets = 0
                    if len(pts) > 0:
                        output = draw_points(img, pts, color='red', size=10)
                        output.save(os.path.join(dest_plots, img_name), quality=95)

                        # Create and export thumbnails
                        sp_score = list(img_dets[['species','scores']].to_records(index=False))
                        for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score)):
                            off = args.ts//2
                            coords = (x - off, y - off, x + off, y + off)
                            thumbnail = img_cpy.crop(coords)
                            score = round(score * 100, 0)
                            thumbnail = draw_text(thumbnail, f"{sp} | {score}%", 
                                                position=(10,5), font_size=int(0.08*args.ts))
                            thumbnail.save(os.path.join(dest_thumb, img_name[:-4] + f'_{i}.JPG'))
                            n_dets += 1
                    
                    del img, img_cpy
                    return n_dets
                except Exception as e:
                    print(f'[WARNING] Failed to process visualization for {img_name}: {e}')
                    return 0
            
            # Process images in parallel using thread pool
            total_exported = 0
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_single_image, img_name): img_name 
                          for img_name in batch_img_names}
                for future in as_completed(futures):
                    total_exported += future.result()
            
            print(f'[BATCH {batch_start//args.batch + 1}] Exported {total_exported} detection thumbnails')
            
        except Exception as e:
            print(f'[ERROR] Batch {batch_start}-{batch_end} failed: {e}')
            import traceback
            traceback.print_exc()
        
        # Clear memory after this batch
        del evaluator_batch, metrics_batch, dataloader_batch, dataset_batch, df_batch
        if 'detections_batch' in locals():
            del detections_batch
        
        clear_memory()
        print_memory_usage("BATCH END")
        print(f'[BATCH {batch_start//args.batch + 1}] Completed and memory cleared')
    
    # Final summary
    print("\n" + "="*80)
    print("INFERENCE COMPLETED")
    print("="*80)
    
    if all_detections:
        total_detections = sum(len(d) for d in all_detections)
        print(f"\nTotal detections: {total_detections}")
        print(f"Results saved in: {dest}")
        print(f"Detections CSV: {detections_path}")
        print(f"Plots folder: {dest_plots}")
        print(f"Thumbnails folder: {dest_thumb}")
    else:
        print("\nWARNING: No detections were generated")
    
    print_memory_usage("FINAL")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()


