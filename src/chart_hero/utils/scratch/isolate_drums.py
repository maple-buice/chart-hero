import os
import pathlib

from mido import MidiFile


def getChartFiles(directory, fileName):
    chartFiles = []
    for path, _, files in os.walk(directory):
        for name in files:
            if name == fileName:
                chartFiles.append(pathlib.PurePath(path, name))
    return chartFiles


def getDrumNotesFromCharts(chartFiles):
    drumNotes = []
    for chartFile in chartFiles:
        with open(chartFile, "r") as file:
            lines = file.readlines()
            foundDrums = False
            for line in lines:
                if not foundDrums:
                    if line.startswith("[ExpertDrums]"):
                        foundDrums = True
                else:
                    if line.startswith("{"):
                        continue
                    if line.startswith("}"):
                        break
                    drumNotes.append(line)
    return drumNotes


def getNoteCountsFromCharts(drumNotes):
    noteCounts: dict[int, int] = {}
    for note in drumNotes:
        noteValue = note.split(" ")[5]
        if not noteValue.isnumeric():
            continue
        noteInt = int(noteValue)
        if noteInt in noteCounts:
            noteCounts[noteInt] += 1
        else:
            noteCounts[noteInt] = 1
    return dict(sorted(noteCounts.items()))


def getDrumNotesFromMidis(midiFiles):
    drumNotes = []
    for file in midiFiles:
        try:
            with MidiFile(file) as midiFile:
                for track in midiFile.tracks:
                    if "drum" not in track.name.lower():
                        continue
                    print(track.name)
                    for message in track:
                        if message.type == "note_on":
                            drumNotes.append(message.note)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    return drumNotes


def getNoteCountsFromMidis(drumNotes):
    noteCounts: dict[int, int] = {}
    for noteInt in drumNotes:
        if noteInt in noteCounts:
            noteCounts[noteInt] += 1
        else:
            noteCounts[noteInt] = 1
    return dict(sorted(noteCounts.items()))


songsRoot = "/Users/maple/Clone Hero/Songs"
# chartFiles = getChartFiles(songsRoot, "notes.chart")
# drumNotesFromCharts = getDrumNotesFromCharts(chartFiles)
# noteCountsFromCharts = getNoteCountsFromCharts(drumNotesFromCharts)

# print(noteCountsFromCharts)

# open("drum_notes.txt", "w").write("".join(drumNotes))

midiFiles = getChartFiles(songsRoot, "notes.mid")
drumNotesFromMidis = getDrumNotesFromMidis(midiFiles)
noteCountsFromMidis = getNoteCountsFromMidis(drumNotesFromMidis)

print(noteCountsFromMidis)
