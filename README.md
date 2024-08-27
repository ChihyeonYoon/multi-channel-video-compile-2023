# multi-channel-video-compile-2023

## Directory Structure
```
|-- multi-channel-video-compile-2023
    |-- requirements.txt
    
    |-- compiled_sample
        |-- sample_opentalk.json # 전체구간 추론 결과
        |-- sample_thelive.json # 전체구간 추론 결과
    |-- compile_materials
        |-- opentalk
            |-- camera1_synced.mp4
            |-- camera2_synced.mp4
            |-- camera3_synced.mp4
            |-- label.csv
        |-- thelive
            |-- C.mp4
            |-- D.mp4
            |-- MC.mp4
            |-- W.mp4
            |-- label.csv
    
    |-- face_inference.py
    |-- speaking_detection_module.py
    |-- speaking_detection_model_weight.pth

    |-- video_channel_inference.py
    |-- video_channel_compile.py
    |-- video_inference_and_compile.sh
    
    |-- video_channel_inference_the_live.py
    |-- video_channel_compile_the_live.py
    |-- video_inference_and_compile_the_live.sh
```