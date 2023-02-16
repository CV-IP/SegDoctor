
# cityscapes
CUDA_VISIBLE_DEVICES="0" torchrun \
  --nproc_per_node=1 \
  --master_port='29515' \
    train_seg.py \
  --use_ddp \
  --sync_bn \
  --num_classes 19 \
  --batch_size 8 \
  --epochs 100 \

# VOC
# CUDA_VISIBLE_DEVICES="0" torchrun \
#   --nproc_per_node=2 \
#   --master_port='29507' \
#     finetune_seg.py \
#   --use_ddp \
#   --sync_bn \
#   --num_classes 21 \
#   --batch_size 8 \

