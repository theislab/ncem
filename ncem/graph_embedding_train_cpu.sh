python torch_models/non_linear_ncem/train.py  \
--latent_dim=30 \
--num_workers=8 \
--batch_size=1 \
--lr=0.01 \
--weight_decay=0.0 \
--check_val_every_n_epoch=5 \
--log_every_n_steps=10 \
--max_epochs=1000 \
# --encoder_hidden_dims 30 30 \
# --accelerator=gpu \
# --gradient_clip_val=0.1 \