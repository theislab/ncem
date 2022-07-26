python torch_models/graph_embedding/train.py \
--latent_dim=40 \
--num_workers=12 \
--batch_size=1 \
--accelerator=gpu \
--lr=0.01 \
--weight_decay=0.001 \
--check_val_every_n_epoch=5 \
--log_every_n_steps=10 \
--max_epochs=1000 \
# --encoder_hidden_dims 30 30 \