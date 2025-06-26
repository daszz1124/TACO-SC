import os
import json
import matplotlib.pyplot as plt

# 后端设置（避免 show() 报错）
import matplotlib
matplotlib.use('Agg')  # 使用无 GUI 的后端

# 根目录
base_dir = './compression_mscoco_val30k'

# 初始化列表
bps, psnrs, ms_ssims, lpips_list = [], [], [], []

# 遍历子目录
for subdir in sorted(os.listdir(base_dir)):
    stat_path = os.path.join(base_dir, subdir, 'mean_stat.json')
    if os.path.exists(stat_path):
        with open(stat_path, 'r') as f:
            stat = json.load(f)
            bps.append(stat['bpp'])
            psnrs.append(stat['psnr'])
            ms_ssims.append(stat['ms_ssim'])
            lpips_list.append(stat['lpips'])

# 按 bpp 排序
sorted_data = sorted(zip(bps, psnrs, ms_ssims, lpips_list))
bps, psnrs, ms_ssims, lpips_list = zip(*sorted_data)

# 创建图像保存目录
save_dir = './rd_plots'
os.makedirs(save_dir, exist_ok=True)

# -------- PSNR 曲线 --------
plt.figure()
plt.plot(bps, psnrs, marker='o', color='blue')
plt.xlabel('Bit per Pixel (bpp)')
plt.ylabel('PSNR (dB)')
plt.title('RD Curve: PSNR vs BPP')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rd_psnr.png'))
plt.close()

# -------- MS-SSIM 曲线 --------
plt.figure()
plt.plot(bps, ms_ssims, marker='s', color='green')
plt.xlabel('Bit per Pixel (bpp)')
plt.ylabel('MS-SSIM')
plt.title('RD Curve: MS-SSIM vs BPP')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rd_msssim.png'))
plt.close()

# -------- LPIPS 曲线 --------
plt.figure()
plt.plot(bps, lpips_list, marker='^', color='red')
plt.xlabel('Bit per Pixel (bpp)')
plt.ylabel('LPIPS')
plt.title('RD Curve: LPIPS vs BPP')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rd_lpips.png'))
plt.close()

print(f"✅ 图像已保存至 {save_dir}")
