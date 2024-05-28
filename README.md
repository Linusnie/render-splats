
# render-splats

Notebook with minimal setup for rendering the .ply output from gaussian splatting

# Setup

```bash
python -m pip install torch # modify to match your cuda version if necessary
python -m pip install plyfile pycolmap matplotlib ipykernel ipympl gsplat pyyaml
python -m pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
```

Depending on your CUDA version you might have to build gsplat manually as described in the [readme](https://github.com/nerfstudio-project/gsplat/blob/main/README.md)
