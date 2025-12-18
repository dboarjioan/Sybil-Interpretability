import subprocess
import os
import torch
import numpy as np
from SybilInference.src_sybil.sybil.serie import Serie
from SybilInference.src_sybil.sybil.models.sybil import SybilNet

resultTrain = subprocess.run(
    ['python3', 'MedGS/train.py', '-s', 'data/lungs', '-m', 'output/lungs']
)
resultRender = subprocess.run(
    ['python3', 'MedGS/render.py', '--model_path', 'output/lungs', '--interp', '1']
)

png_folder = "output/lungs/render"
model_path = "28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt"

voxel_spacing = [1.0, 1.0, 1.0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

png_files = sorted(
    os.path.join(png_folder, f)
    for f in os.listdir(png_folder)
    if f.endswith(".png")
)

if len(png_files) == 0:
    raise ValueError(f"No PNG files found in folder: {png_folder}")

serie = Serie(
    dicoms = png_files,
    voxel_spacing = voxel_spacing,
    file_type = "png",
    split = "test"
)

model = SybilNet.load(model_path)
model.to(device)
model.eval()

volume = serie.get_volume().to(device)

with torch.no_grad():
    output = model(volume)
    risck_score = output["prob"]

print(risck_score)

# resultVideoMedGS = subprocess.run(
#     ['python3', 'MedGS/video.py', '--input_folder', 'output/lungs/render', '--output_folder', 'video', '--output_name', 'lungs_medgs.mp4']
# )

# resultVideo = subprocess.run(
#     ['python3', 'MedGS/video.py', '--input_folder', 'data/lungs/original', '--output_folder', 'video', '--output_name', 'lungs.mp4']
# )
#export PYTHONPATH=$PYTHONPATH:$(pwd)/SybilInference/src_sybil
