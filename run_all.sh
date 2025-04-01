datasets=("MSL" "PSM" "SMAP" "SWAT")
# datasets=("MSL" "PSM" "SMAP" "WADI" "MBA")
# datasets=("MSL" "PSM" "MBA")
# datasets=("SWAT" "SMD" "WADI")

datasets=("MBA" "exathlon" "SMD" "WADI")
# datasets=("MBA")
train_anomaly_ratio=0.00
noise_step=100
seq_len=100
pred_len=100
win_size=100
step=$pred_len

AD_model=AT
joint_epochs=5
cross_attn_epochs=5
ftr_idx=0
SYN_INJ=""
ONLY_SYN_INJ=""
ADD_AN_F=""
ADD_AN_AD=""
SYN_INJ="--synthetic_injection"
ONLY_SYN_INJ="--only_synthetic_injection"
SHARE="--share"


d_model=384
for cross_attn_epochs in 5; do
  for cross_attn_nhead in 1; do
    for dataset in "${datasets[@]}"; do
      job_name="${dataset}_ours_${d_model}"
      # job_name="${dataset}_ours_noAAFN_minmax"
      log_file="logs/ICML/scalability/${dataset}/${job_name}_%j_%N.out"

      mkdir -p logs/ICML/scalability/$dataset
      
      sbatch --job-name="$job_name" \
            --gres=gpu:1 \
            --nodes=1 \
            -w ariel-g2 \
            --cpus-per-gpu=8 \
            --mem-per-gpu=20G \
            --time=6-0 \
            --partition=batch_grad \
            -o "$log_file" \
            --wrap="
            conda activate PatchTST
            python -u run_ablation_hp.py \
            --root_path /local_datasets/AD_datasets/$dataset --dataset $dataset \
            --model_id F+AD_${seq_len}_${pred_len} \
            --seq_len $seq_len --pred_len $pred_len --win_size $win_size \
            --step $step --noise_step $noise_step \
            --joint_epochs $joint_epochs \
            --train_anomaly_ratio $train_anomaly_ratio $SHARE \
            --AD_model $AD_model \
            --noise_injection \
            --pretrain_noise \
            --synthetic_injection \
            --only_synthetic_injection \
            --contrastive_loss \
            --cross_attn --cross_attn_epochs $cross_attn_epochs \
            --cross_attn_nheads $cross_attn_nhead --ftr_idx $ftr_idx \
             --aafn_amplify --amplify_type 5type --learnable \
             --d_model $d_model
            # --scaler minmax
            "
    done
  done
done
# python data_provider/multivariate_generator.py
# for node in "ariel-g1" "ariel-g2" "ariel-g3" "ariel-g4" "ariel-g5"; do
#     sbatch --job-name=dataset \
#            --gres=gpu:1 \
#            --nodes=1 \
#            -w "$node" \
#            --cpus-per-gpu=8 \
#            --mem-per-gpu=20G \
#            --time=6-0 \
#            --partition=batch_grad \
#            --wrap="rm -rf /local_datasets/AD_datasets/
#            rm -rf /local_datasets/data/
#            cd /data/pmy0792/dataset
#            tar -xvf AD_datasets.tar -C /local_datasets
#            cd /data/pmy0792/repo/neurips24/ours
#            python data_provider/multivariate_generator.py"
# done