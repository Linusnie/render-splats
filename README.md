
# render-splats

Notebook with minimal setup for rendering the .ply output from gaussian splatting

Should exactly match the output of render.py from the original implementation ([here](https://github.com/graphdeco-inria/gaussian-splatting))

# Setup

```bash
python -m pip install torch # modify to match your cuda version if necessary
python -m pip install plyfile pycolmap matplotlib ipykernel ipympl
python -m pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
```
