video_ch0_path="compile_materials/opentalk/camera1_synced.mp4"
video_ch1_path="compile_materials/opentalk/camera2_synced.mp4"
video_ch2_path="compile_materials/opentalk/camera3_synced.mp4"

start_time=180
end_time=300 # 1분 ~ 3분 구간 -> start_time=60, end_time=180 (단위: 초)
switching_interval=15 # fps=30 이므로, 15 -> 0.5초마다 스위칭 

sample_name="sample_opentalk" # sample_opentalk.json 파일을 사용

inference_result_dict_path="compiled_sample/$sample_name.json"
output_video_path="compiled_sample/$sample_name.mp4" 
# output video filename will be {$sample_name}_compiletime.mp4

model_path="speaking_detection_model_weight.pth"

sleep 1s

if [ -f $inference_result_dict_path ]; then
    echo "$inference_result_dict_path already exists. skip inference"

elif [ -f $video_ch0_path ] && [ -f $video_ch1_path ] && [ -f $video_ch2_path ]; then
    python video_channel_inference.py --video_ch0_path $video_ch0_path \
    --video_ch1_path $video_ch1_path \
    --video_ch2_path $video_ch2_path \
    --inference_result_dict_path $inference_result_dict_path \
    --model_path $model_path \
    \
    # --start_time $start_time \
    # --end_time $end_time \
    # 위 두 옵션을 주석처리하면 전체 영상에 대한 inference를 수행합니다.
fi

sleep 3s

if [ -f $inference_result_dict_path ] && [ -f $video_ch0_path ] && [ -f $video_ch1_path ] && [ -f $video_ch2_path ]; then
    python video_channel_compile.py --video_ch0_path $video_ch0_path \
    --video_ch1_path $video_ch1_path \
    --video_ch2_path $video_ch2_path \
    --switching_interval $switching_interval \
    --inference_result_dict_path $inference_result_dict_path \
    --output_video_path $output_video_path \
    \
    --start_time $start_time \
    --end_time $end_time \
    # 위 두 옵션을 주석처리하면 전체 영상에 대한 compile을 수행합니다.
else
    echo "inference result dict not found. or video files not found."
fi