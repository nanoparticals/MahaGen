import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem

def write_poscar(mol, filename, box_size=30.0):
    """
    将 RDKit 分子对象写入 VASP POSCAR 格式。
    分子将被置于 box_size * box_size * box_size 的盒子中心。
    """
    # 1. 加氢并生成 3D 构象
    mol = Chem.AddHs(mol)
    # 尝试使用 ETKDG 算法生成 3D 结构
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if res != 0:
        # 如果失败，尝试随机坐标
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res != 0:
            print(f"Error: Failed to embed molecule for {filename}")
            return False

    # 2. 获取坐标并进行中心化处理
    conf = mol.GetConformer()
    coords = conf.GetPositions()

    # 计算分子几何中心
    centroid = np.mean(coords, axis=0)

    # 计算平移向量：将分子中心移动到盒子中心 (box_size/2, box_size/2, box_size/2)
    box_center = np.array([box_size / 2.0] * 3)
    shift = box_center - centroid
    new_coords = coords + shift

    # 3. 整理原子数据（按元素分组，因为 VASP 要求相同元素聚在一起）
    atom_data = []
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        pos = new_coords[i]
        atom_data.append({'symbol': symbol, 'pos': pos})
    
    # 按元素符号排序 (例如：C 在前，H 在后)
    atom_data.sort(key=lambda x: x['symbol'])
    
    # 统计每种元素的原子数量
    elements = []     # 存储元素符号列表，如 ['C', 'H', 'O']
    counts = []       # 存储对应数量，如 [6, 12, 1]
    current_sym = ""
    
    if atom_data:
        current_sym = atom_data[0]['symbol']
        elements.append(current_sym)
        counts.append(0)
        
        for atom in atom_data:
            sym = atom['symbol']
            if sym != current_sym:
                elements.append(sym)
                counts.append(0)
                current_sym = sym
            counts[-1] += 1
        
    # 4. 写入文件
    with open(filename, 'w') as f:
        # 第一行：注释行
        f.write(f"Molecule_{os.path.basename(filename)}\n")
        # 第二行：缩放系数
        f.write("1.0\n")
        # 第三-五行：晶胞矢量 (Lattice Vectors)
        f.write(f"{box_size:.6f} 0.000000 0.000000\n")
        f.write(f"0.000000 {box_size:.6f} 0.000000\n")
        f.write(f"0.000000 0.000000 {box_size:.6f}\n")
        # 第六行：元素种类
        f.write(" ".join(elements) + "\n")
        # 第七行：原子数量
        f.write(" ".join(map(str, counts)) + "\n")
        # 第八行：坐标格式
        f.write("Cartesian\n")
        # 后续行：原子坐标
        for atom in atom_data:
            f.write(f"{atom['pos'][0]:.6f} {atom['pos'][1]:.6f} {atom['pos'][2]:.6f}\n")
            
    print(f"Successfully generated: {filename}")
    return True

# --- 主程序 ---

# 1. 读取 CSV 文件
input_csv = '/public/home/users/haoxw/generate_AI/branch_OLED/ckpts/gpu_rl_vgae_soft_congen_16/outputs/all_molecules_dedup.csv' # 确保文件名与你的CSV一致
output_dir = '/public/home/users/haoxw/generate_AI/branch_OLED/DFT/poscar_files'

if not os.path.exists(input_csv):
    print(f"Error: {input_csv} not found.")
else:
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    # 2. 提取前 10 个分子
    # 假设 CSV 中有一列叫 'smiles'，如果列名不同请修改此处
    top_molecules = df['smiles'].head(10).tolist()

    print(f"Found {len(top_molecules)} molecules. Starting conversion...")

    # 3. 循环生成 POSCAR
    for i, smiles in enumerate(top_molecules):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # 命名文件为 POSCAR_01, POSCAR_02 ...
            filename = os.path.join(output_dir, f"POSCAR_{i+1:02d}")
            write_poscar(mol, filename, box_size=30.0)
        else:
            print(f"Warning: Invalid SMILES at index {i}: {smiles}")

    print("Done! Check the 'poscar_files' folder.")

import os
import shutil
import subprocess

# ================= 配置区域 =================
# 根目录
BASE_DIR = "/public/home/users/haoxw/generate_AI/branch_OLED/DFT"
# POSCAR 所在的文件夹 (上一步生成的)
POSCAR_SOURCE_DIR = os.path.join(BASE_DIR, "poscar_files")
# 计算工作目录 (将要创建的)
CALC_DIR = os.path.join(BASE_DIR, "dft_calculate_PBEO_16")
# 需要复制的模板文件 (确保这些文件在 BASE_DIR 下存在)
FILES_TO_COPY = ["INCAR", "vasp.sh"] 
# 注意：KPOINTS 和 POTCAR 我们用 vaspkit 自动生成，所以不需要复制

# ===========================================

def run_command(cmd, cwd=None, input_str=None):
    """辅助函数：执行 Shell 命令"""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            input=input_str, 
            text=True, 
            shell=True, 
            capture_output=True
        )
        if result.returncode != 0:
            print(f"Error running command: {cmd}")
            print(result.stderr)
        return result
    except Exception as e:
        print(f"Exception: {e}")

def main():
    # 1. 确保目标大文件夹存在
    if not os.path.exists(CALC_DIR):
        os.makedirs(CALC_DIR)
        print(f"Created main directory: {CALC_DIR}")

    # 2. 遍历 1 到 10 (假设文件名是 POSCAR_01, POSCAR_02...)
    for i in range(1, 11):
        task_name = f"task_{i:02d}"  # 文件夹名: task_01, task_02...
        task_path = os.path.join(CALC_DIR, task_name)
        
        # 2.1 创建任务文件夹
        if not os.path.exists(task_path):
            os.makedirs(task_path)
        
        print(f"Processing {task_name}...")

        # 2.2 复制并重命名 POSCAR
        # 假设源 POSCAR 名字是 POSCAR_01, POSCAR_02...
        src_poscar = os.path.join(POSCAR_SOURCE_DIR, f"POSCAR_{i:02d}")
        dst_poscar = os.path.join(task_path, "POSCAR")
        
        if os.path.exists(src_poscar):
            shutil.copy2(src_poscar, dst_poscar)
        else:
            print(f"Warning: {src_poscar} not found! Skipping.")
            continue

        # 2.3 复制 INCAR, vasp.sh
        for filename in FILES_TO_COPY:
            src_file = os.path.join(BASE_DIR, filename)
            dst_file = os.path.join(task_path, filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: Template file {filename} missing in {BASE_DIR}")

        # 2.4 调用 vaspkit 生成 KPOINTS (Gamma 1x1x1)
        # 对应 vaspkit 功能 102 (Generate K-Points Mesh) -> 2 (Gamma-Centered Mesh) -> 1 (1x1x1)
        # 我们使用 input 参数模拟键盘输入
        # 输入序列: "102" (功能) -> "2" (Gamma Scheme) -> "1" (1x1x1 grid)
        # 注意：不同版本 vaspkit 菜单略有不同，这是最通用的
        
        # 为了稳妥，对于分子计算，其实写死一个 KPOINTS 文件最安全
        # 这里用 Python 直接写一个标准的 Gamma 点 KPOINTS，比调用 vaspkit 更快更稳
        kpoints_content = """Gamma-point only
0
Gamma
1 1 1
0 0 0
"""
        with open(os.path.join(task_path, "KPOINTS"), "w") as f:
            f.write(kpoints_content)
        # print(f"  - Generated KPOINTS (Gamma 1x1x1)")

        # 2.5 调用 vaspkit 生成 POTCAR (功能 103/104)
        # 假设你已经配置好了 ~/.vaspkit 文件里的 POTCAR 路径
        # 自动生成 POTCAR: 输入 "103" (或者你版本里的 Generate POTCAR)
        # vaspkit 1.x 版本通常是 功能 103 (Generate POTCAR)
        
        # 尝试调用 vaspkit 生成 POTCAR
        # 输入: "103" (功能)
        run_command("vaspkit -task 103", cwd=task_path)
        
        # 检查 POTCAR 是否生成成功
        if not os.path.exists(os.path.join(task_path, "POTCAR")):
             print(f"  - Warning: POTCAR generation failed in {task_path}. Check vaspkit configuration.")
        else:
             print(f"  - POTCAR generated.")

        # 2.6 (可选) 提交任务
        run_command("qsub vasp.sh", cwd=task_path)

    print("\nAll setup done! Go to output directory and check files.")

if __name__ == "__main__":
    main()