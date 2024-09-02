video_ch0_path="compile_materials/thelive/W.mp4"
video_ch1_path="compile_materials/thelive/C.mp4"
video_ch2_path="compile_materials/thelive/D.mp4"
video_chMC_path="compile_materials/thelive/MC.mp4"

start_time=0
end_time=180 # 1분 ~ 3분 구간 -> start_time=60, end_time=180 (단위: 초)
switching_interval=15 # fps=30 이므로, 15 -> 0.5초마다 스위칭 

sample_name="sample_thelive" # sample_thelive.json 파일을 사용

inference_result_dict_path="compiled_sample/$sample_name.json"
output_video_path="compiled_sample/$sample_name.mp4" 
# output video filename will be {$sample_name}_compiletime.mp4

model_path="speaking_detection_model_weight.pth"

sleep 1s

if [ -f $inference_result_dict_path ]; then
    echo "$inference_result_dict_path already exists. skip inference"

elif [ -f $video_ch0_path ] && [ -f $video_ch1_path ] && [ -f $video_ch2_path ] && [ -f $video_chMC_path ]; then
    python video_channel_inference_the_live.py --video_ch0_path $video_ch0_path \
    --video_ch1_path $video_ch1_path \
    --video_ch2_path $video_ch2_path \
    --video_chMC_path $video_chMC_path \
    --inference_result_dict_path $inference_result_dict_path \
    --model_path $model_path \
    \
    # --start_time $start_time \
    # --end_time $end_time \
    # 위 두 옵션을 주석처리하면 전체 영상에 대한 inference를 수행합니다.
fi

sleep 3s

if [ -f $inference_result_dict_path ] && [ -f $video_ch0_path ] && [ -f $video_ch1_path ] && [ -f $video_ch2_path ] && [ -f $video_chMC_path ]; then
    python video_channel_compile_the_live.py --video_ch0_path $video_ch0_path \
    --video_ch1_path $video_ch1_path \
    --video_ch2_path $video_ch2_path \
    --video_chMC_path $video_chMC_path \
    --switching_interval $switching_interval \
    --inference_result_dict_path $inference_result_dict_path \
    --output_video_path $output_video_path \
    \
    --start_time $start_time \
    --end_time $end_time \
    # 위 두 옵션을 주석처리하면 전체 영상에 대한 compile을 수행합니다.

else
    echo "inference result dict not found."
fi