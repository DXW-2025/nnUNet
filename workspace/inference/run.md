nnUNetv2_predict \
  -i /root/autodl-tmp/nnUNet/workspace/inference/905_0_tse/input_unet \
  -o /root/autodl-tmp/nnUNet/workspace/inference/905_0_tse/output \
  -d 905 -c 2d -f 0 -chk checkpoint_best.safetensors
