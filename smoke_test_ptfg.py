import sys, torch
from contsg.config.schema import ExperimentConfig
from contsg.models.pt_factor_generator import PTFactorGeneratorModule

print("=== PTFactorGenerator smoke test ===")

cfg_m = ExperimentConfig.from_yaml("configs/generators/ptfg_synth-m.yaml")
m = PTFactorGeneratorModule(cfg_m, use_condition=True)
p = sum(v.numel() for v in m.parameters() if v.requires_grad)
print(f"synth-m  params={p:,}")
batch_m = {"ts": torch.randn(4,128,2), "attrs": torch.randint(0,2,(4,4))}
out = m(batch_m)
print(f"  train loss={out['loss'].item():.4f}")
samp = m.generate(torch.randint(0,2,(4,4)), n_samples=2, steps=2)
print(f"  generate={samp.shape}  expected=(4,2,128,2)")

cfg_u = ExperimentConfig.from_yaml("configs/generators/ptfg_synth-u.yaml")
mu = PTFactorGeneratorModule(cfg_u, use_condition=True)
pu = sum(v.numel() for v in mu.parameters() if v.requires_grad)
print(f"synth-u  params={pu:,}")
batch_u = {"ts": torch.randn(4,128,1), "cap_emb": torch.randn(4,1024)}
out2 = mu(batch_u)
print(f"  train loss={out2['loss'].item():.4f}")
samp2 = mu.generate(torch.randn(4,1024), n_samples=2, steps=2)
print(f"  generate={samp2.shape}  expected=(4,2,128,1)")

print("ALL OK")
sys.exit(0)
