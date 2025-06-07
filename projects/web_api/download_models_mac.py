import os
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    # 设置安全的本地保存路径（当前用户目录下）
    base_local_dir = os.path.expanduser("~/.cache/pdf-extract-kit")
    layoutreader_local_dir = os.path.expanduser("~/.cache/layoutreader")

    os.makedirs(base_local_dir, exist_ok=True)
    os.makedirs(layoutreader_local_dir, exist_ok=True)

    # 只下载 PDF-Extract-Kit 所需的特定子目录
    mineru_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_small_2501/*",
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    model_dir = snapshot_download(
        "opendatalab/PDF-Extract-Kit-1.0",
        allow_patterns=mineru_patterns,
        local_dir=base_local_dir,
        cache_dir=base_local_dir,  # 显式指定 cache_dir，避免访问 /opt
        resume_download=True
    )

    # 下载 layoutreader 模型
    layoutreader_pattern = ["*.json", "*.safetensors"]
    layoutreader_model_dir = snapshot_download(
        "hantian/layoutreader",
        allow_patterns=layoutreader_pattern,
        local_dir=layoutreader_local_dir,
        cache_dir=layoutreader_local_dir,
        resume_download=True
    )

    # 输出模型路径
    model_dir = os.path.join(model_dir, "models")
    print(f"✅ model_dir is: {model_dir}")
    print(f"✅ layoutreader_model_dir is: {layoutreader_model_dir}")