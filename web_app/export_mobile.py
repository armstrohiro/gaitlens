"""Export the fixed research checkpoint; verify ONNX parity on a public fixture."""
from pathlib import Path
import hashlib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from foot23_models import extend_model
from pose_pipeline import model_paths, PROJECT

class ExportView(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.network = model

    def forward(self, x):
        y = self.network(x)
        return y[0] if isinstance(y, tuple) else y

def main():
    paths = model_paths()
    model = extend_model(paths['yolo_base'], 'cpu')
    model.load_state_dict(torch.load(paths['yolo_proposed'], map_location='cpu', weights_only=True)['state_dict'])
    model.eval()
    head = model.model[-1]
    head.export = True
    head.format = 'onnx'
    head.dynamic = False
    wrapper = ExportView(model).eval()
    fixture = PROJECT / 'results/foot23_browser_probe_20261025_v1/selftest/input.f32'
    x = torch.from_numpy(np.fromfile(fixture, dtype='<f4').reshape(1, 3, 640, 640).copy())
    output = Path(__file__).parent / 'models' / 'tug_all_frames_epoch5.onnx'
    output.parent.mkdir(exist_ok=True)
    with torch.inference_mode():
        reference = wrapper(x).numpy()
        torch.onnx.export(wrapper, x, str(output), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['detections'], do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    result = session.run(None, {'images': x.numpy()})[0]
    if reference.shape != (1, 74, 8400) or result.shape != reference.shape:
        raise ValueError(f'予期しない出力形状: {reference.shape}, {result.shape}')
    difference = float(np.max(np.abs(reference - result)))
    if difference >= .05:
        raise ValueError(f'出力差が許容値を超えました: {difference}')
    print('output:', output)
    print('sha256:', hashlib.sha256(output.read_bytes()).hexdigest())
    print('raw output shape:', result.shape, 'max_abs_difference:', difference)

if __name__ == '__main__':
    main()

