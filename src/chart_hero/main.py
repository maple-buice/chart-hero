import argparse
from os import path
import os
from chart_hero.inference.song_identifier import get_data_from_acousticbrainz, identify_song
from chart_hero.inference.input_transform import drum_extraction, drum_to_frame, get_yt_audio
from chart_hero.inference.charter import drum_charter
import librosa
import torch
import pandas as pd
from chart_hero.model_training.train_transformer import DrumTranscriptionModule

def predict_with_transformer(model, drum_track, sample_rate, config):
    """
    Predict drum hits using the trained transformer model.
    """
    # The model expects a spectrogram, so we need to create one from the drum track.
    # The drum_to_frame function already creates audio clips, so we can use those.
    df, bpm = drum_to_frame(drum_track, sample_rate)
    
    # Create a SpectrogramProcessor to create spectrograms from the audio clips
    from chart_hero.model_training.transformer_data import SpectrogramProcessor
    processor = SpectrogramProcessor(config)
    
    # Create a list of spectrograms
    spectrograms = []
    for i in range(len(df)):
        audio_clip = df.audio_clip.iloc[i]
        spectrogram = processor.audio_to_spectrogram(torch.from_numpy(audio_clip).float().unsqueeze(0))
        spectrograms.append(spectrogram.unsqueeze(0))
        
    # Stack the spectrograms into a batch
    spectrograms = torch.cat(spectrograms)
    
    # Make predictions
    with torch.no_grad():
        model.eval()
        predictions = model(spectrograms)
        
    # Process predictions
    preds = torch.sigmoid(predictions['logits']) > 0.5
    
    # Create a DataFrame with the predictions
    from chart_hero.model_training.transformer_config import get_drum_hits
    drum_hits = get_drum_hits()
    prediction_df = pd.DataFrame(preds.numpy(), columns=drum_hits)
    
    # Combine with the original DataFrame
    df.reset_index(inplace=True)
    prediction_df.reset_index(inplace=True)
    result = df.merge(prediction_df,left_on='index', right_on= 'index')
    result.drop(columns=['index'],inplace=True)
    
    return result, bpm

def main():
    parser = argparse.ArgumentParser(description="Transcribe the drum part of a given song", usage=None)

    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument('-l', '--link',
                        type=str,
                        help='Youtube video link')
    
    input.add_argument('-p', '--path',
                        type=str,
                        help='Path to local audio file')

    parser.add_argument('-km', '--kernel_mode',
                        choices=['performance', 'speed'],
                        type=str,
                        required=True,
                        help="The processing mode of the kernel, either speed or performance. "
                                "Speed mode is 4 times faster than performance mode but quality could be slightly worse")

    parser.add_argument('-bpm',
                        default=None,
                        type=int,
                        help='The estimated bpm of the song')

    parser.add_argument('-r', '--resolution',
                        default=16,
                        choices=[None, 4,8,16,32],
                        help='Control the window size (total length) of the onset sound clip extract from the song')

    parser.add_argument('-b', '--beat',
                        type=int,
                        default=4,
                        help='Number of beats in each measure')
        
    parser.add_argument('-n', '--note',
                        type=int,
                        default=4,
                        help="The UPPER NUMBER of the song's time signature." 
                                "This number represent the number of beats in each measure.")

    parser.add_argument('-fmt', '--format',
                        default='pdf',
                        choices=['pdf', 'musicxml'],
                        type=str,
                        help='Output sheet music format')
    
    parser.add_argument('-o', '--outpath',
                        default='',
                        type=str,
                        help='Output sheet music directory path')
                        
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model checkpoint')

    args = parser.parse_args()

    if args.link!=None:
        print(f'Downloading audio track from {args.link}')
        f_path = get_yt_audio(args.link)
        print(f'Audio track saved to {f_path}')
    else:
        f_path=args.path
        print(f'Retriving audio track from {args.path}')
    
    print('Getting song info')
    song_info = identify_song(f_path)
    
    acousticbrainz_info = get_data_from_acousticbrainz(song_info)
    
    with open('acousticbrainz_result.json', 'w') as f:
        f.write(acousticbrainz_info.__str__())
    
    print('Start Demixing Process...')
    drum_track, sample_rate = drum_extraction(f_path,
                                              mode=args.kernel_mode)

    print('Drum track extracted')

    print('Loading model...')
    from chart_hero.model_training.transformer_config import get_config
    config = get_config('local') # Use a default config
    model = DrumTranscriptionModule.load_from_checkpoint(args.model_path, config=config)

    print('Converting drum track and making predictions...')
    df_pred, bpm = predict_with_transformer(model, drum_track, sample_rate, config)

    print('Creating chart...')

    song_duration = librosa.get_duration(y=drum_track, sr=sample_rate)

    sheet_music = drum_charter(df_pred,
                                    song_duration,
                                    bpm,
                                    sample_rate,
                                    beats_in_measure=args.beat,
                                    note_value=args.note)
    
    if args.format=='pdf':
        out_path=sheet_music.sheet.write(fmt='musicxml.pdf', fp=os.path.join(args.outpath, song_info['title']))
        print(f'Sheet music saved at {out_path}')
    else:
        out_path= sheet_music.sheet.write(fp=os.path.join(args.outpath, song_info['title']))
        print(f'Sheet music saved at {out_path}')
    if args.link!=None:
        os.remove(f_path)
if __name__ == "__main__":
    main()