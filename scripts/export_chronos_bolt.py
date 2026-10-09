"""Xuất Chronos-Bolt tiny sang ONNX cho dự báo nhu cầu đặt sân (chạy ở stage build riêng có PyTorch).

    python scripts/export_chronos_bolt.py <thư mục ra>

Ra ``chronos-bolt-tiny.onnx`` + ``chronos-bolt-tiny.json``. Chỉ xuất phần mạng nơ-ron; chuẩn hóa / chia đoạn / chạy nối
nằm ở ``app/infrastructure/forecast/chronos_onnx.py``. Script dừng (exit 1) nếu module đó lệch pipeline PyTorch gốc,
nên image chỉ được build khi bản ONNX + numpy cho đúng kết quả của Chronos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from chronos import BaseChronosPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.infrastructure.forecast.chronos_onnx import (  # noqa: E402
    ChronosBoltConfig,
    ChronosBoltForecaster,
)

MODEL_ID = "amazon/chronos-bolt-tiny"
# Cố định commit trên Hugging Face: đổi mô hình là đổi dòng này (và chạy lại kiểm tra bên dưới).
REVISION = "a0e552de83495b5c28c14c71c374f3e33280b340"
TOLERANCE = 1e-4


class _Core(torch.nn.Module):
    """``ChronosBoltModelForForecasting.forward`` bỏ tiền / hậu xử lý (ONNX không hỗ trợ nanmean, unfold động)."""

    def __init__(self, model) -> None:
        super().__init__()
        self.m = model

    def forward(self, patched_context, attention_mask):
        m = self.m
        batch = patched_context.shape[0]
        embeds = m.input_patch_embedding(patched_context)
        if m.chronos_config.use_reg_token:
            reg = m.shared.weight[m.config.reg_token_id].view(1, 1, -1).expand(batch, 1, -1)
            embeds = torch.cat([embeds, reg], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1)
        hidden = m.encoder(attention_mask=attention_mask, inputs_embeds=embeds)[0]
        start = m.shared.weight[m.config.decoder_start_token_id].view(1, 1, -1).expand(batch, 1, -1)
        decoded = m.decoder(
            inputs_embeds=start, encoder_hidden_states=hidden, encoder_attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        return m.output_patch_embedding(decoded).view(batch, len(m.chronos_config.quantiles), -1)


def _test_series() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    hours = np.arange(24 * 7 * 10)
    daily = 0.15 + 0.6 * np.exp(-((hours % 24 - 19) ** 2) / 8) + 0.1 * ((hours // 24) % 7 >= 5)
    weekly = np.clip(daily + rng.normal(0, 0.05, hours.size), 0, 1).astype(np.float32)
    gappy = weekly[-24 * 21:].copy()
    gappy[:30] = np.nan
    sparse = np.zeros(24 * 30, np.float32)
    sparse[rng.choice(sparse.size, 12, replace=False)] = 0.5
    return {"10 tuần": weekly, "3 tuần có NaN": gappy, "thưa": sparse, "1 ngày": weekly[-24:]}


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, revision=REVISION, device_map="cpu", torch_dtype=torch.float32)
    model = pipeline.model.eval()
    cc = model.chronos_config
    if cc.input_patch_stride != cc.input_patch_size or cc.context_length % cc.input_patch_size:
        raise SystemExit("Cấu hình Chronos-Bolt khác giả định (patch stride / context length).")
    config = ChronosBoltConfig(
        context_length=cc.context_length,
        patch_size=cc.input_patch_size,
        prediction_length=cc.prediction_length,
        quantiles=tuple(float(q) for q in cc.quantiles),
        use_arcsinh=bool(model.instance_norm.use_arcsinh),
        eps=float(model.instance_norm.eps),
        model_name="chronos-bolt-tiny",
        revision=REVISION,
    )

    patches = config.context_length // config.patch_size
    dummy = (torch.zeros(1, patches, 2 * config.patch_size), torch.ones(1, patches))
    onnx_path = out_dir / "chronos-bolt-tiny.onnx"
    torch.onnx.export(
        _Core(model).eval(), dummy, str(onnx_path),
        input_names=["patched_context", "attention_mask"], output_names=["quantile_preds"],
        dynamic_axes={"patched_context": {0: "batch"}, "attention_mask": {0: "batch"}, "quantile_preds": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx_path.with_suffix(".json").write_text(
        json.dumps({**config.__dict__, "quantiles": list(config.quantiles)}, indent=2), encoding="utf-8"
    )

    forecaster = ChronosBoltForecaster(onnx_path)
    worst = 0.0
    for name, series in _test_series().items():
        for horizon in (config.prediction_length, 168):
            with torch.no_grad():
                reference = pipeline.predict(torch.tensor(series)[None], prediction_length=horizon).numpy()[0]
            diff = float(np.abs(forecaster.predict(series, horizon) - reference).max())
            worst = max(worst, diff)
            print(f"{name:>14} · {horizon:>3} giờ · lệch tối đa {diff:.2e}")
    if worst > TOLERANCE:
        raise SystemExit(f"Bản ONNX lệch PyTorch {worst:.2e} > {TOLERANCE}: không dùng được.")
    print(f"OK {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), revision {REVISION}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
