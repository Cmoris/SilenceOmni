import json
from inference import SilenceDemoInfer
from dataclasses import dataclass, field
import transformers

@dataclass
class ModelArguments:
    num_beams : int = field(default=None)
    temperature : float = field(default=None)
    model_path : str = field(default=None) 
    model_base : str = field(default=None) 
    model_type : str = field(default="SilenceQwen3")

if __name__ == '__main__':
    video_path = "/n/work1/muyun/SemiAutonomous/videofiles/20240124_01_C_audio_video_subject.mp4"
    audio_path = "/n/work1/muyun/SemiAutonomous/wavefiles/20240124_01_C_audio_subject.wav"
    query = """Please analyze the silent section at the end of the video 
            and determine which of the following two categories best describes it: 
            {class: 0, content: stopped} {class: 1, content: thinking} 
            Return your answer as a single JSON object, 
            choosing only one of the two categories above."""
    parser = transformers.HfArgumentParser(ModelArguments)
    model_args,  = parser.parse_args_into_dataclasses()
    model_args.model_path = "/n/work1/muyun/Model/SilenceStreaming/Qwen2.5Omni/checkpoint-1212"
    model_args.model_base = "Qwen/Qwen2.5-Omni-3B"
    infer = SilenceDemoInfer(model_args=model_args)
    state = {'video_path': video_path, 'audio_path': audio_path}
    commentaries = []
    t = 0
    for t in range(1000):
        state['video_timestamp'] = t
        state['audio_timestamp'] = t
        for (start_t, stop_t), response, state in infer.live_cc(
            message=query, state=state, 
            repetition_penalty=1.05, 
            streaming_eos_base_threshold=None, 
            streaming_eos_threshold_step=0
        ):
            print(f'{start_t}s-{stop_t}s: {response}')
            commentaries.append([start_t, stop_t, response])
        if state.get('video_end', False):
            break

    result = {'video_path': video_path, 'query': query, 'commentaries': commentaries}
    result_path = video_path.replace('/assets/', '/results/').replace('.mp4', '.json')
    print(f"{video_path=}, {query=} => {model_args.model_path=} => {result_path=}")
    json.dump(result, open(result_path, 'w'))