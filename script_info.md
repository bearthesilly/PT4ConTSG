# Dataset
-bash-4.2 pwd
/storage/group/renkan/lansc/gents_bench/huggingface_archive/datasets
-bash-4.2 ls -R .
.:
airquality_beijing  ettm1  istanbul_traffic  ptbxl_concept  ptbxl_morphology  synth-m  synth-u  telecomts_segment  weather_concept  weather_morphology

./airquality_beijing:
meta.json                                                train_attrs_idx.npy                                       valid_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy
test_attrs_idx.npy                                       train_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy  valid_text_caps_embeddings_qwen3-embedding-0.6b.npy
test_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy  train_text_caps_embeddings_qwen3-embedding-0.6b.npy       valid_text_caps.npy
test_text_caps_embeddings_qwen3-embedding-0.6b_64.npy    train_text_caps.npy                                       valid_ts.npy
test_text_caps.npy                                       train_ts.npy
test_ts.npy                                              valid_attrs_idx.npy

./ettm1:
meta.json           test_cap_emb.npy    test_ts.npy          train_cap_emb.npy    train_ts.npy         valid_cap_emb.npy    valid_ts.npy
test_attrs_idx.npy  test_text_caps.npy  train_attrs_idx.npy  train_text_caps.npy  valid_attrs_idx.npy  valid_text_caps.npy

./istanbul_traffic:
meta.json           test_cap_emb.npy    test_ts.npy          train_cap_emb.npy    train_ts.npy         valid_cap_emb.npy    valid_ts.npy
test_attrs_idx.npy  test_text_caps.npy  train_attrs_idx.npy  train_text_caps.npy  valid_attrs_idx.npy  valid_text_caps.npy

./ptbxl_concept:
meta.json                 test_attrs_idx.npy  test_text_caps.npy  train_attrs_idx.npy  train_text_caps.npy  valid_attrs_idx.npy  valid_text_caps.npy
normalization_stats.json  test_cap_emb.npy    test_ts.npy         train_cap_emb.npy    train_ts.npy         valid_cap_emb.npy    valid_ts.npy

./ptbxl_morphology:
meta.json                 test_attrs_idx.npy  test_text_caps.npy  train_attrs_idx.npy  train_text_caps.npy  valid_attrs_idx.npy  valid_text_caps.npy
normalization_stats.json  test_cap_emb.npy    test_ts.npy         train_cap_emb.npy    train_ts.npy         valid_cap_emb.npy    valid_ts.npy

./synth-m:
meta.json                                                train_attrs_idx.npy                                       valid_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy
test_attrs_idx.npy                                       train_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy  valid_text_caps_embeddings_qwen3-embedding-0.6b.npy
test_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy  train_text_caps_embeddings_qwen3-embedding-0.6b.npy       valid_text_caps.npy
test_text_caps_embeddings_qwen3-embedding-0.6b_64.npy    train_text_caps.npy                                       valid_ts.npy
test_text_caps.npy                                       train_ts.npy
test_ts.npy                                              valid_attrs_idx.npy

./synth-u:
meta.json                                                train_attrs_idx.npy                                       valid_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy
test_attrs_idx.npy                                       train_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy  valid_text_caps_embeddings_qwen3-embedding-0.6b.npy
test_text_caps_embeddings_qwen3-embedding-0.6b_1024.npy  train_text_caps_embeddings_qwen3-embedding-0.6b.npy       valid_text_caps.npy
test_text_caps_embeddings_qwen3-embedding-0.6b_64.npy    train_text_caps.npy                                       valid_ts.npy
test_text_caps.npy                                       train_ts.npy
test_ts.npy                                              valid_attrs_idx.npy

./telecomts_segment:
meta.json                 test_cap_emb.npy        test_ts.npy          train_caps.npy           valid_attrs_idx.npy  valid_segment_caps.json
normalization_stats.json  test_caps.npy           train_attrs_idx.npy  train_segment_caps.json  valid_cap_emb.npy    valid_ts.npy
test_attrs_idx.npy        test_segment_caps.json  train_cap_emb.npy    train_ts.npy             valid_caps.npy

./weather_concept:
meta.json                test_attrs_idx.npy  test_caps.npy  train_attrs_idx.npy  train_caps.npy  valid_attrs_idx.npy  valid_caps.npy
normalization_stats.npz  test_cap_emb.npy    test_ts.npy    train_cap_emb.npy    train_ts.npy    valid_cap_emb.npy    valid_ts.npy

./weather_morphology:
meta.json                test_attrs_idx.npy  test_caps.npy  train_attrs_idx.npy  train_caps.npy  valid_attrs_idx.npy  valid_caps.npy
normalization_stats.npz  test_cap_emb.npy    test_ts.npy    train_cap_emb.npy    train_ts.npy    valid_cap_emb.npy    valid_ts.npy

# CTTP config yaml and ckpt

airquality_beijing_intrinsic (legacy 格式)
* config: /storage/group/renkan/lansc/gents_bench/save/airquality_beijing_intrinsic_cttp/model_configs.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/save/airquality_beijing_intrinsic_cttp/clip_model_best.pth
ettm1_llm
* config: /storage/group/renkan/lansc/gents_bench/contsg_repo/configs/ettm1_llm_clip/cttp_ettm1_llm_instance_ce.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/constg_results/20260113_173113_ettm1_llm_cttp_ce/checkpoints/finetune/finetune-epoch=35-val/loss=3.3886.ckpt
istanbul_traffic_llm
* config: /storage/group/renkan/lansc/gents_bench/contsg_repo/configs/istanbul_traffic_llm_clip/cttp_istanbul_traffic_llm_instance_ce.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/contsg_repo/experiments/20260113_173113_istanbul_traffic_llm_cttp_ce/20260113_173113_istanbul_traffic_llm_cttp_ce/checkpoints/finetune/finetune-epoch=84-val/loss=3.2515.ckpt
ptb_extrinsic
* config: /storage/group/renkan/lansc/gents_bench/contsg_repo/experiments/ptb_extrinsic_cttp/20251230_172132_ptbxl_extrinsic_cttp/config.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/constg_results/ptb_extrinsic_cttp/20251230_172132_ptbxl_extrinsic_cttp/checkpoints/finetune/finetune-epoch=145-val/loss=4.0224.ckpt
ptb_intrinsic
* config: /storage/group/renkan/lansc/gents_bench/contsg_repo/experiments/ptb_intrinsic_cttp/20251230_172328_ptbxl_intrinsic_cttp/config.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/constg_results/ptb_intrinsic_cttp/20251230_172328_ptbxl_intrinsic_cttp/checkpoints/finetune/finetune-epoch=183-val/loss=4.1135.ckpt
synth-m (legacy 格式)
* config: /storage/group/renkan/lansc/gents_bench/save/synth-m_cttp/model_configs.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/save/synth-m_cttp/clip_model_best.pth
synth-u (legacy 格式)
* config: /storage/group/renkan/lansc/gents_bench/save/synth-u_cttp/model_configs.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/save/synth-u_cttp/clip_model_best.pth
telecomts_segment
* config: /storage/group/renkan/lansc/gents_bench/contsg_repo/configs/telecomts_clip/cttp_telecomts_instance_ce.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/contsg_repo/experiments/cttp_telecomts_ce/cttp_telecomts_ce/20260110_194154_telecomts_segment_cttp/checkpoints/finetune/finetune-epoch=53-val/loss=1.9437.ckpt
weather_extrinsic
* config: /storage/group/renkan/lansc/gents_bench/contsg_repo/configs/weather_clip_extrinsic_intrinsic/cttp_weather_extrinsic_ce.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/constg_results/20260112_112337_weather_extrinsic_cttp_ce/checkpoints/finetune/finetune-epoch=31-val/loss=2.9512.ckpt
weather_intrinsic
* config: /storage/group/renkan/lansc/gents_bench/contsg_repo/configs/weather_clip_extrinsic_intrinsic/cttp_weather_intrinsic_ce.yaml
* checkpoint: /storage/group/renkan/lansc/gents_bench/constg_results/20260112_201827_weather_intrinsic_cttp_ce/checkpoints/finetune/finetune-epoch=58-val/loss=2.9277.ckpt