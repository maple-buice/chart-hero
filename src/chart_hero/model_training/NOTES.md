# Notes about past models and runs

## First full training session: `full_midi_model.keras`

Labels were full midi map:

```python
return [
    22, # Hi-hat Closed (Edge)
    26, # Hi-hat Open (Edge)
    35, # Acoustic Bass Drum
    36, # Kick / Bass Drum 1
    37, # Snare X-Stick / Side Stick
    38, # Snare (Head) / Acoustic Snare
    39, # Hand Clap	/ Cowbell
    40, # Snare (Rim) / Electric Snare
    41, # Low Floor Tom
    42, # Hi-hat Closed (Bow) / Closed Hi-Hat
    43, # Tom 3 (Head) / High Floor Tom
    44, # Hi-hat Pedal / Pedal Hi-Hat
    45, # Tom 2 / Low Tom
    46, # Hi-hat Open (Bow) / Open Hi-Hat
    47, # Tom 2 (Rim) / Low-Mid Tom
    48, # Tom 1 / Hi-Mid Tom
    49, # Crash 1 (Bow) / Crash Cymbal 1
    50, # Tom 1 (Rim) / High Tom
    51, # Ride (Bow) / Ride Cymbal 1
    52, # Crash 2 (Edge) / Chinese Cymbal
    53, # Ride (Bell) / Ride Bell
    54, # Tambourine / Cowbell
    55, # Crash 1 (Edge) / Splash Cymbal
    56, # Cowbell
    57, # Crash 2 (Bow) / Crash Cymbal 2
    58, # Tom 3 (Rim) / Vibraslap
    59, # Ride (Edge) / Ride Cymbal 2
    60, # Hi Bongo
    61, # Low Bongo
    62, # Mute Hi Conga
    63, # Open Hi Conga
    64, # Low Conga
    65, # High Timbale
    66, # Low Timbale
    67, # High Agogo
    68, # Low Agogo
    69, # Cabasa
    70, # Maracas
    71, # Short Whistle
    72, # Long Whistle
    73, # Short Guiro
    74, # Long Guiro
    75, # Claves
    76, # Hi Wood Block
    77, # Low Wood Block
    78, # Mute Cuica
    79, # Open Cuica
    80, # Mute Triangle
    81, # Open Triangle
]
```

Results of analysis classification report:

| Instrument | **note** | **precision** | **recall** | **f1-score** | **support** |
|-:|:-:|:-:|:-:|:-:|:-:|
| Hi-hat Closed (Edge) | 22 | 0.75 | 0.79 | 0.77 | 9732 |
| Hi-hat Open (Edge) | 26 | 0.75 | 0.67 | 0.71 | 2562 |
| Acoustic Bass Drum | 35 | 0.00 | 0.00 | 0.00 | 0 |
| Kick / Bass Drum 1 | 36 | 0.68 | 0.84 | 0.75 | 28768 |
| Snare X-Stick / Side Stick | 37 | 0.70 | 0.76 | 0.72 | 3137 |
| Snare (Head) / Acoustic Snare | 38 | 0.78 | 0.89 | 0.83 | 38486 |
| Hand Clap	/ Cowbell | 39 | 0.78 | 0.67 | 0.72 | 103 |
| Snare (Rim) / Electric Snare | 40 | 0.62 | 0.90 | 0.73 | 7299 |
| Low Floor Tom | 41 | 0.00 | 0.00 | 0.00 | 0 |
| Hi-hat Closed (Bow) / Closed Hi-Hat | 42 | 0.77 | 0.82 | 0.80 | 11606 |
| Tom 3 (Head) / High Floor Tom | 43 | 0.69 | 0.82 | 0.75 | 3475 |
| Hi-hat Pedal / Pedal Hi-Hat | 44 | 0.74 | 0.61 | 0.67 | 14357 |
| Tom 2 / Low Tom | 45 | 0.60 | 0.65 | 0.62 | 1028 |
| Hi-hat Open (Bow) / Open Hi-Hat | 46 | 0.77 | 0.38 | 0.50 | 831 |
| Tom 2 (Rim) / Low-Mid Tom | 47 | 0.55 | 0.46 | 0.50 | 250 |
| Tom 1 / Hi-Mid Tom | 48 | 0.81 | 0.82 | 0.82 | 4121 |
| Crash 1 (Bow) / Crash Cymbal 1 | 49 | 0.00 | 0.00 | 0.00 | 312 |
| Tom 1 (Rim) / High Tom | 50 | 0.70 | 0.45 | 0.55 | 436 |
| Ride (Bow) / Ride Cymbal 1 | 51 | 0.70 | 0.82 | 0.76 | 13058 |
| Crash 2 (Edge) / Chinese Cymbal | 52 | 0.51 | 0.11 | 0.18 | 283 |
| Ride (Bell) / Ride Bell | 53 | 0.74 | 0.56 | 0.64 | 1235 |
| Tambourine / Cowbell | 54 | 0.67 | 0.75 | 0.71 | 1845 |
| Crash 1 (Edge) / Splash Cymbal | 55 | 0.51 | 0.50 | 0.51 | 677 |
| Cowbell | 56 | 0.92 | 0.82 | 0.87 | 28 |
| Crash 2 (Bow) / Crash Cymbal 2 | 57 | 0.00 | 0.00 | 0.00 | 17 |
| Tom 3 (Rim) / Vibraslap | 58 | 0.70 | 0.50 | 0.59 | 244 |
| Ride (Edge) / Ride Cymbal 2 | 59 | 0.61 | 0.43 | 0.50 | 745 |
| Hi Bongo | 60 | 0.00 | 0.00 | 0.00 | 0 |
| Low Bongo | 61 | 0.00 | 0.00 | 0.00 | 0 |
| Mute Hi Conga | 62 | 0.00 | 0.00 | 0.00 | 0 |
| Open Hi Conga | 63 | 0.00 | 0.00 | 0.00 | 0 |
| Low Conga | 64 | 0.00 | 0.00 | 0.00 | 0 |
| High Timbale | 65 | 0.00 | 0.00 | 0.00 | 0 |
| Low Timbale | 66 | 0.00 | 0.00 | 0.00 | 0 |
| High Agogo | 67 | 0.00 | 0.00 | 0.00 | 0 |
| Low Agogo | 68 | 0.00 | 0.00 | 0.00 | 0 |
| Cabasa | 69 | 0.00 | 0.00 | 0.00 | 0 |
| Maracas | 70 | 0.00 | 0.00 | 0.00 | 0 |
| Short Whistle | 71 | 0.00 | 0.00 | 0.00 | 0 |
| Long Whistle | 72 | 0.00 | 0.00 | 0.00 | 0 |
| Short Guiro | 73 | 0.00 | 0.00 | 0.00 | 0 |
| Long Guiro | 74 | 0.00 | 0.00 | 0.00 | 0 |
| Claves | 75 | 0.00 | 0.00 | 0.00 | 0 |
| Hi Wood Block | 76 | 0.00 | 0.00 | 0.00 | 0 |
| Low Wood Block | 77 | 0.00 | 0.00 | 0.00 | 0 |
| Mute Cuica | 78 | 0.00 | 0.00 | 0.00 | 0 |
| Open Cuica | 79 | 0.00 | 0.00 | 0.00 | 0 |
| Mute Triangle | 80 | 0.00 | 0.00 | 0.00 | 0 |
| Open Triangle | 81 | 0.00 | 0.00 | 0.00 | 0 |
| --------- | ---- | --------- | ----- | --------- | -------- |
| **micro** | avg | 0.73 | 0.80 | 0.76 | 144635 |
| **macro** | avg | 0.33 | 0.31 | 0.31 | 144635 |
| **weighted** | avg | 0.73 | 0.80 | 0.76 | 144635 |
| **samples** | avg | 0.73 | 0.73 | 0.73 | 144635 |
