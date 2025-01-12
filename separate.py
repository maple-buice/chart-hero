import os
import demucs.separate
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

# demucs.separate.main(
#     ["--mp3",
#      "-n", "htdemucs_6s",
#      "-j", "32",
#      "music/Chappell Roan - HOT TO GO.mp3"])

musicDir = "music"
songFileName = "Imagine Dragons - Believer.mp3"
songFileNameNoExtension = songFileName.replace(".mp3", "")

demucsOutputDir = "separated"
demucsModel = "htdemucs_6s"
demucsSplitSongOutputDir = "/Users/maple/src/clone-hero-hero/" + demucsOutputDir + "/" + demucsModel + "/" + songFileNameNoExtension

if os.path.exists(demucsSplitSongOutputDir):
    print("Song already split, moving on to MIDI generation")
else:
    demucs.separate.main(
        ["--mp3",
        "-o", demucsOutputDir,
        "-n", demucsModel,
        "-j", "32",
        musicDir + "/" + songFileName])

basicPitchOutputDir = "midi_output"
basicPitchSongOutputDir = "/Users/maple/src/clone-hero-hero/" + basicPitchOutputDir + "/" + songFileNameNoExtension

if not os.path.exists(basicPitchOutputDir):
    os.makedirs(basicPitchOutputDir)
    
# if os.path.exists(basicPitchSongOutputDir):
#     print("MIDI already generated, moving on to next song")
# else:
# os.makedirs(basicPitchSongOutputDir)
predict_and_save(
    [demucsSplitSongOutputDir + "/drums.mp3"], # absoluteFilePaths(demucsSplitSongOutputDir),
    basicPitchSongOutputDir, 
    True,
    False,
    False,
    True,
    ICASSP_2022_MODEL_PATH,
    0.25, 
    0.15, 
    50, 
    None, 
    None, 
    False,
)
