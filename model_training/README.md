# Drum Note Mapping

*Model training code adapted from https://github.com/cb-42/AnNOTEator, which uses the MIT license*

The MIDI note mapping from the original AnNOTEator was:
```python
self.midi_note_map={36: 'KD', 38: 'SD', 40: 'SD', 37: 'SD_xstick', 48: 'HT', 50: 'HT',
                   45: 'MT', 47: 'MT', 43: 'FT' ,58: 'FT', 46: 'HH_open', 
                   26: 'HH_open', 42: 'HH_close', 22: 'HH_close', 44: 'HH_close',
                   49: 'CC', 57: 'CC', 55: 'CC', 52: 'CC', 51: 'RC',
                   59: 'RC', 53: 'RC', 39: 'CB', 54: 'CB', 56: 'CB'}
```

Which was based on this mapping table from Google's Groove MIDI Dataset (the expanded version of which is used for training data):

| Pitch | Roland Mapping | GM Mapping | Paper Mapping | Frequency |
|-------|----------------|------------|---------------|-----------|
| 22 | HH Closed (Edge) | N/A | Closed Hi-Hat (42) | 34764 |
| 26 | HH Open (Edge) | N/A | Open Hi-Hat (46) | 10243 |
| 36 | Kick | Bass Drum 1 | Bass (36) | 88067 |
| 37 | Snare X-Stick | Side Stick | Snare (38) | 9696 |
| 38 | Snare (Head) | Acoustic Snare | Snare (38) | 102787 |
| 40 | Snare (Rim) | Electric Snare | Snare (38) | 22262 |
| 42 | HH Closed (Bow) | Closed Hi-Hat | Closed Hi-Hat (42) | 31691 |
| 43 | Tom 3 (Head) | High Floor Tom | High Floor Tom (43) | 11260 |
| 44 | HH Pedal | Pedal Hi-Hat | Closed Hi-Hat (42) | 52343 |
| 45 | Tom 2 | Low Tom | Low-Mid Tom (47) | 3935 |
| 46 | HH Open (Bow) | Open Hi-Hat | Open Hi-Hat (46) | 3905 |
| 47 | Tom 2 (Rim) | Low-Mid Tom | Low-Mid Tom (47) | 1322 |
| 48 | Tom 1 | Hi-Mid Tom | High Tom (50) | 13145 |
| 49 | Crash 1 (Bow) | Crash Cymbal 1 | Crash Cymbal (49) | 720 |
| 50 | Tom 1 (Rim) | High Tom | High Tom (50) | 1561 |
| 51 | Ride (Bow) | Ride Cymbal 1 | Ride Cymbal (51) | 43847 |
| 52 | Crash 2 (Edge) | Chinese Cymbal | Crash Cymbal (49) | 1046 |
| 53 | Ride (Bell) | Ride Bell | Ride Cymbal (51) | 5567 |
| 55 | Crash 1 (Edge) | Splash Cymbal | Crash Cymbal (49) | 5567 |
| 57 | Crash 2 (Bow) | Crash Cymbal 2 | Crash Cymbal (49) | 1832 |
| 58 | Tom 3 (Rim) | Vibraslap | High Floor Tom (43) | 1003 |
| 59 | Ride (Edge) | Ride Cymbal 2 | Ride Cymbal (51) | 2220 |

## Learnings from MoonScraper and CloneHero

### MoonScraper offset constants

Offsets to get from fret to special note type
    - c_proDrumsOffset gets us to cymbal
    - c_instrumentPlusOffset gets us to double kick

```csharp
public const int c_proDrumsOffset = 64;
public const int c_instrumentPlusOffset = 32;
public const int c_drumsAccentOffset = 33;
public const int c_drumsGhostOffset = 39;

// Note velocities
public const byte VELOCITY = 100;             // default note velocity for exporting
public const byte VELOCITY_ACCENT = 127;      // fof/ps
public const byte VELOCITY_GHOST = 1;         // fof/ps
```

### Clone Hero MIDI drum map 

*Copied from https://wiki.clonehero.net/books/guitars-drums-controllers/page/midi-drums*

| Mapping       |   Drums           |   Lefty Flip      |
| ------------- | ----------------- | ----------------- |
| Red Pad       |   Snare           |   Low Tom         |
| Yellow Pad    |   High Tom        |   Middle Tom      |
| Blue Pad      |   Middle Tom      |   High Tom        |
| Green Pad     |   Low Tom         |   Snare           |
| Orange Pad    |   Kick            |   Kick            |
| Yellow Cymbal |   Hi-hat Cymbal   |   Ride Cymbal     |
| Blue Cymbal   |   Ride Cymbal     |   Hi-Hat Cymbal   |
| Green Cymbal  |   Crash Cymbal    |   Crash Cymbal    |

Which means that the cymbal to pad mapping would be (C# enum formatting for readability):

```csharp
public enum CymbalPad
{
    // Assign to the sprite array position
    CrashCymbal = 64,  // (DrumPad.Green -> GuitarFret.Orange -> 4) + 64,
    // Red = 1,
    HiHatCymbal = 65, // (DrumPad.Yellow -> GuitarFret.Red -> 1) + 64,
    RideCymbal = 66,  // (DrumPad.Blue -> GuitarFret.Yellow -> 2) + 64,
    // Orange = 4,
    // Open = 5
}
```

### MoonScraper DrumsChartNoteNumberToProcessFnMap
```csharp
static readonly IReadOnlyDictionary<int, NoteEventProcessFn> DrumsChartNoteNumberToProcessFnMap = new Dictionary<int, NoteEventProcessFn>()
{
    { 0, (int)Note.DrumPad.Kick }, // Kick
    { 1, (int)Note.DrumPad.Red }, // Snare
    { 2, (int)Note.DrumPad.Yellow }, // High tom
    { 3, (int)Note.DrumPad.Blue }, // Middle tom
    { 4, (int)Note.DrumPad.Orange }, // Low tom, automatically converted from Green to Orange in 4 lane mode
    { 5, (int)Note.DrumPad.Green }, // Low tom, functionally 4 (not 5)

    { ChartIOHelper.c_instrumentPlusOffset, (int)Note.DrumPad.Kick, Note.Flags.DoubleKick }, // 32

    { ChartIOHelper.c_proDrumsOffset + 2, (int)Note.DrumPad.Yellow, NoteFlagPriority.Cymbal }, // Hi-hat, 66
    { ChartIOHelper.c_proDrumsOffset + 3, (int)Note.DrumPad.Blue, NoteFlagPriority.Cymbal }, // Ride, 67
    { ChartIOHelper.c_proDrumsOffset + 4, (int)Note.DrumPad.Orange, NoteFlagPriority.Cymbal }, // Crash, 68

    // { ChartIOHelper.c_drumsAccentOffset + 0, ... }  // Reserved for kick accents, if they should ever be a thing
    { ChartIOHelper.c_drumsAccentOffset + 1, (int)Note.DrumPad.Red, NoteFlagPriority.Accent }, // 34
    { ChartIOHelper.c_drumsAccentOffset + 2, (int)Note.DrumPad.Yellow, NoteFlagPriority.Accent }, // 35
    { ChartIOHelper.c_drumsAccentOffset + 3, (int)Note.DrumPad.Blue, NoteFlagPriority.Accent }, // 36
    { ChartIOHelper.c_drumsAccentOffset + 4, (int)Note.DrumPad.Orange, NoteFlagPriority.Accent }, // 37
    { ChartIOHelper.c_drumsAccentOffset + 5, (int)Note.DrumPad.Green, NoteFlagPriority.Accent }, // 38

    // { ChartIOHelper.c_drumsGhostOffset + 0, ... }  // Reserved for kick ghosts, if they should ever be a thing
    { ChartIOHelper.c_drumsGhostOffset + 1, (int)Note.DrumPad.Red, NoteFlagPriority.Ghost }, // 40
    { ChartIOHelper.c_drumsGhostOffset + 2, (int)Note.DrumPad.Yellow, NoteFlagPriority.Ghost }, // 41
    { ChartIOHelper.c_drumsGhostOffset + 3, (int)Note.DrumPad.Blue, NoteFlagPriority.Ghost }, // 42
    { ChartIOHelper.c_drumsGhostOffset + 4, (int)Note.DrumPad.Orange, NoteFlagPriority.Ghost }, // 43 (actually Green)
    { ChartIOHelper.c_drumsGhostOffset + 5, (int)Note.DrumPad.Green, NoteFlagPriority.Ghost }, // 44
};
```

Taking into account the counts from all chart files I've downloaded:
```python
{
    0: 38292, 
    1: 32640, 
    2: 47563, 
    3: 13492, 
    4: 12420, 
    32: 2089, 
    34: 842, 
    35: 1983, 
    36: 558, 
    37: 111, 
    40: 5504, 
    41: 510, 
    42: 179, 
    43: 830, 
    64: 1209, 
    65: 58, 
    66: 41932, 
    67: 10834, 
    68: 7209
}
```

And the full enum map could be simplified to:

```csharp
public enum ProDrumMap {
    Kick = 0,
    Snare = 1,
    HighTom = 2,
    MiddleTom = 3,
    LowTom = 4,
    CrashCymbal = 66,
    HiHatCymbal = 67,
    RideCymbal = 68,
}
```

Count of all the drum note positions in downloaded `notes.chart` files, as a sanity check:

```json
{
    "0": 38292,
    "1": 32640,
    "2": 47563,
    "3": 13492,
    "4": 12420,
    "32": 2089,
    "34": 842,
    "35": 1983,
    "36": 558,
    "37": 111,
    "40": 5504,
    "41": 510,
    "42": 179,
    "43": 830,
    "64": 1209,
    "65": 58,
    "66": 41932,
    "67": 10834,
    "68": 7209
}
```

Hi-tom seems very high, but maybe it's skewed in the dataset in the `notes.chart` dataset.

Here it is for the MIDIs:

```json
```

## Final Map

Combining the Google mapping, the AnNOTEator mapping, and the general MIDI mapping from [ZenDrum](https://www.zendrum.com/resource-site/drumnotes.htm) (for good measure), we get:

```python
self.midi_note_map={
    22: '67', # Hi-hat Closed (Edge) -> HiHatCymbal
    26: '67', # Hi-hat Open (Edge) -> HiHatCymbal
    35: '0', # Acoustic Bass Drum -> Kick
    36: '0', # Kick / Bass Drum 1 -> Kick
    37: '1', # Snare X-Stick / Side Stick -> Snare
    38: '1', # Snare (Head) / Acoustic Snare -> Snare
    39: '67', # Hand Clap	/ Cowbell -> HiHatCymbal
    40: '1', # Snare (Rim) / Electric Snare -> Snare
    41: '4', # Low Floor Tom	-> LowTom
    42: '67', # Hi-hat Closed (Bow) / Closed Hi-Hat -> HiHatCymbal
    43: '4', # Tom 3 (Head) / High Floor Tom -> LowTom
    44: '67', # Hi-hat Pedal / Pedal Hi-Hat -> HiHatCymbal
    45: '3', # Tom 2 / Low Tom -> MiddleTom
    46: '67', # Hi-hat Open (Bow) / Open Hi-Hat -> HiHatCymbal
    47: '3', # Tom 2 (Rim) / Low-Mid Tom -> MiddleTom
    48: '2', # Tom 1 / Hi-Mid Tom -> HighTom
    49: '66', # Crash 1 (Bow) / Crash Cymbal 1 -> CrashCymbal
    50: '2', # Tom 1 (Rim) / High Tom -> HighTom
    51: '68', # Ride (Bow) / Ride Cymbal 1 -> RideCymbal
    52: '66', # Crash 2 (Edge) / Chinese Cymbal -> CrashCymbal
    53: '68', # Ride (Bell) / Ride Bell -> RideCymbal
    54: '67', # Tambourine / Cowbell -> HiHatCymbal
    55: '66', # Crash 1 (Edge) / Splash Cymbal -> CrashCymbal
    56: '67', # Cowbell -> HiHatCymbal
    57: '66', # Crash 2 (Bow) / Crash Cymbal 2 -> CrashCymbal
    58: '4', # Tom 3 (Rim) / Vibraslap -> LowTom
    59: '68', # Ride (Edge) / Ride Cymbal 2 -> RideCymbal
    60: '2', # Hi Bongo -> HighTom
    61: '3', # Low Bongo -> MiddleTom
    62: '2', # Mute Hi Conga -> HighTom
    63: '3', # Open Hi Conga -> MiddleTom
    64: '4', # Low Conga -> LowTom
    65: '2', # High Timbale -> HighTom
    66: '3', # Low Timbale -> MiddleTom
    67: '2', # High Agogo -> HighTom
    68: '3', # Low Agogo -> MiddleTom
    69: '67', # Cabasa -> HiHatCymbal
    70: '67', # Maracas -> HiHatCymbal
    71: '68', # Short Whistle -> RideCymbal
    72: '66', # Long Whistle -> CrashCymbal
    73: '68', # Short Guiro -> RideCymbal
    74: '66', # Long Guiro -> CrashCymbal
    75: '67', # Claves -> HiHatCymbal
    76: '2', # Hi Wood Block -> HighTom
    77: '3', # Low Wood Block -> MiddleTom
    78: '2', # Mute Cuica -> HighTom
    79: '3', # Open Cuica -> MiddleTom
    80: '68', # Mute Triangle -> RideCymbal
    81: '66', # Open Triangle -> CrashCymbal
}
```

## Future improvements

It may be possible to train the model to support special note types like accent, ghost, and double kick. I'm going to leave that for a future version.
