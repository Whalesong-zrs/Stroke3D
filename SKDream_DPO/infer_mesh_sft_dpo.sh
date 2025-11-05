# origin skdream mesh generation
export CUDA_VISIBLE_DEVICES=3

MV_DIR=/home/zrs/skdream_model/eval/dpo_sft_dpo/res_mv_1000
COARSE_DIR=/home/zrs/skdream_model/eval/dpo_sft_dpo/res_coarse
REFINE_DIR=/home/zrs/skdream_model/eval/dpo_sft_dpo/res_refine

# LRM reconstruction
# python infer_rec.py config/instant-mesh-large.yaml $MV_DIR --num_view 4 --repeat_num 4 --distance 4.0 \
#     --output_path $COARSE_DIR --save_video --save_img --export_texmap --gpu 0 

# multi-view image tiling, save in the same folder as infer_mv
python infer_tile.py --save_dir $MV_DIR --num_view 4 --repeat_num 4 --gpu 0 

# texture refinement
python infer_refine.py --mesh_dir $COARSE_DIR --tile_dir $MV_DIR --save_dir $REFINE_DIR --repeat_num 4 
