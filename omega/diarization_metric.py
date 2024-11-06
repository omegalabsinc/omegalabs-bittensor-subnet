from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from omega.diarization_pipeline import CustomDiarizationPipeline

def calculate_diarization_metrics(audio_arr, sr, true_segments):
    """Calculate Diarization Error Rate (DER) and related metrics using pyannote metrics"""
    pred_segments = pipeline.process(audio_arr, sr)
    
    # Convert dictionary segments to pyannote Annotation format
    def segments_to_annotation(segments):
        annotation = Annotation()
        for i in range(len(segments['start'])):
            segment = Segment(segments['start'][i], segments['end'][i])
            annotation[segment] = segments['speakers'][i]
        return annotation

    # Convert both predictions and ground truth
    reference = segments_to_annotation(true_segments)
    hypothesis = segments_to_annotation(pred_segments)

    # Calculate metrics using pyannote
    metric = DiarizationErrorRate(skip_overlap=True)
    der = metric(reference, hypothesis)
    # optimal_mapping = metric.optimal_mapping(reference, hypothesis)

    
    # Get detailed components
    components = metric(reference, hypothesis, detailed=True)
    miss_rate = components['missed detection'] / components['total']
    false_alarm_rate = components['false alarm'] / components['total'] 
    speaker_error_rate = components['confusion'] / components['total']

    return {
        "der": 1 - der.clip(0, 1),
        "miss_rate": 1 - miss_rate,
        "false_alarm_rate": 1 - false_alarm_rate,
        "speaker_error_rate": 1 - speaker_error_rate
    }


diarization_model_id = "Revai/reverb-diarization-v2"
overlap_detection_model_id = "pyannote/overlapped-speech-detection" 
pipeline = CustomDiarizationPipeline(overlap_detection_model_id=overlap_detection_model_id,
                                    diarization_model_id=diarization_model_id)
