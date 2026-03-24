# contsg/models/my_model.py
from contsg.models.base import BaseGeneratorModule
from contsg.registry import Registry

@Registry.register_model("my_model")
class MyModelModule(BaseGeneratorModule):
    """My custom generation model."""

    def _build_model(self):
        cfg = self.config.model
        data_cfg = self.config.data
        self.encoder = nn.Linear(cfg.channels, data_cfg.n_var * data_cfg.seq_length)
        self.decoder = nn.Linear(cfg.channels, data_cfg.n_var * data_cfg.seq_length)

    def forward(self, batch):
        ts = batch["ts"]           # (B, L, C) — time series
        cap_emb = batch["cap_emb"] # (B, D)   — text embedding
        # ... compute loss ...
        return {"loss": loss}      # must return dict with "loss" key

    def generate(self, condition, n_samples=1, **kwargs):
        # condition: (B, D) — conditioning tensor
        # return: (B, n_samples, L, C)
        return samples