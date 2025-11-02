DATA_DIR=objsk_eval2
MV_DIR=res_mv
COARSE_DIR=res_coarse
REFINE_DIR=res_refine

# skeletal conditioned multi-view generation
# python infer_mv.py --data_dir $DATA_DIR --save_dir $MV_DIR --num_view 4 --repeat_num 4 --neg_prompt 'default' --cond_scale 1.0 --gpu 0 --mvc_ckpt './ckpt/skdream' 
python infer_mv.py --data_dir $DATA_DIR --save_dir $MV_DIR --num_view 4 --repeat_num 4 --neg_prompt 'default' --cond_scale 1.0 --gpu 0 --mvc_ckpt './ckpt/skdream2' 

# LRM reconstruction
python infer_rec.py config/instant-mesh-large.yaml $MV_DIR --num_view 4 --repeat_num 4 --distance 4.0 \
    --output_path $COARSE_DIR --save_video --save_img --export_texmap --gpu 0 

# multi-view image tiling, save in the same folder as infer_mv
python infer_tile.py --save_dir $MV_DIR --num_view 4 --repeat_num 4 --gpu 0 

# texture refinement
python infer_refine.py --mesh_dir $COARSE_DIR --tile_dir $MV_DIR --save_dir $REFINE_DIR --repeat_num 4 

# calculate alignment score
python eval_align.py --eval_dir objsk_eval/eval.json --img_dir $MV_DIR