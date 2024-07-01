# MultiTHUMOS

This page contains information about the MultiTHUMOS dataset, which was published with the following
paper:

```textmate
Every Moment Counts: Dense Detailed Labeling of Actions in Complex Videos.
Yeung S., Russakovsky O., Jin N., Andriluka M., Mori G., Fei-Fei L.
IJCV 2017 (http://arxiv.org/pdf/1507.05738)
```

The data can be downloaded from
the [official project page](http://ai.stanford.edu/~syyeung/everymoment.html).

## Get the raw data

`arctix` provides the function `arctix.dataset.multithumos.fetch_data` to easily load the raw data
in a `polars.DataFrame` format.

```pycon

>>> from pathlib import Path
>>> from arctix.dataset.multithumos import fetch_data
>>> dataset_path = Path("/path/to/dataset/multithumos")
>>> data_raw = fetch_data(dataset_path)  # doctest: +SKIP
shape: (38_690, 4)
┌──────────────────────────┬────────────┬──────────┬─────────────────────┐
│ video                    ┆ start_time ┆ end_time ┆ action              │
│ ---                      ┆ ---        ┆ ---      ┆ ---                 │
│ str                      ┆ f64        ┆ f64      ┆ str                 │
╞══════════════════════════╪════════════╪══════════╪═════════════════════╡
│ video_test_0000004       ┆ 0.03       ┆ 1.1      ┆ Stand               │
│ video_test_0000004       ┆ 0.2        ┆ 1.1      ┆ CricketBowling      │
│ video_test_0000004       ┆ 0.23       ┆ 0.93     ┆ Jump                │
│ video_test_0000004       ┆ 0.33       ┆ 1.13     ┆ Throw               │
│ video_test_0000004       ┆ 1.0        ┆ 1.5      ┆ CricketShot         │
│ …                        ┆ …          ┆ …        ┆ …                   │
│ video_validation_0000990 ┆ 103.6      ┆ 110.4    ┆ VolleyballSpiking   │
│ video_validation_0000990 ┆ 108.07     ┆ 111.57   ┆ Jump                │
│ video_validation_0000990 ┆ 109.03     ┆ 110.87   ┆ BodyBend            │
│ video_validation_0000990 ┆ 112.9      ┆ 122.47   ┆ TalkToCamera        │
│ video_validation_0000990 ┆ 112.9      ┆ 122.47   ┆ CloseUpTalkToCamera │
└──────────────────────────┴────────────┴──────────┴─────────────────────┘

```

If the data is not downloaded in the dataset path, `fetch_data` automatically downloads the data.
You can set `force_download=True` to force to re-download the data if the data is already
downloaded.

## Prepare the data

`arctix` provides the function `arctix.dataset.multithumos.prepare_data` to preprocess the raw data.
It returns two outputs: the prepared data and the generate metadata.

```pycon

>>> from arctix.dataset.multithumos import prepare_data
>>> data, metadata = prepare_data(data_raw)  # doctest: +SKIP

```

`data` is a `polars.DataFrame` which contains the prepared data:

```textmate
shape: (38_690, 6)
┌─────────────────────────┬────────────┬────────────┬─────────────────────┬───────────┬────────────┐
│ video                   ┆ start_time ┆ end_time   ┆ action              ┆ action_id ┆ split      │
│ ---                     ┆ ---        ┆ ---        ┆ ---                 ┆ ---       ┆ ---        │
│ str                     ┆ f64        ┆ f64        ┆ str                 ┆ i64       ┆ str        │
╞═════════════════════════╪════════════╪════════════╪═════════════════════╪═══════════╪════════════╡
│ video_test_0000004      ┆ 0.03       ┆ 1.1        ┆ Stand               ┆ 3         ┆ test       │
│ video_test_0000004      ┆ 0.2        ┆ 1.1        ┆ CricketBowling      ┆ 32        ┆ test       │
│ video_test_0000004      ┆ 0.23       ┆ 0.93       ┆ Jump                ┆ 1         ┆ test       │
│ video_test_0000004      ┆ 0.33       ┆ 1.13       ┆ Throw               ┆ 4         ┆ test       │
│ video_test_0000004      ┆ 1.0        ┆ 1.5        ┆ CricketShot         ┆ 29        ┆ test       │
│ …                       ┆ …          ┆ …          ┆ …                   ┆ …         ┆ …          │
│ video_validation_000099 ┆ 103.599998 ┆ 110.400002 ┆ VolleyballSpiking   ┆ 37        ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000099 ┆ 108.07     ┆ 111.57     ┆ Jump                ┆ 1         ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000099 ┆ 109.029999 ┆ 110.870003 ┆ BodyBend            ┆ 23        ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000099 ┆ 112.900002 ┆ 122.470001 ┆ TalkToCamera        ┆ 16        ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000099 ┆ 112.900002 ┆ 122.470001 ┆ CloseUpTalkToCamera ┆ 20        ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
└─────────────────────────┴────────────┴────────────┴─────────────────────┴───────────┴────────────┘
```

You can note that new columns are added and the data types of some columns have changed.
`metadata` contains a vocabulary that is used to generate the column `action_id`:

```textmate
{'vocab_action': Vocabulary(
  counter=Counter({'Run': 3693, 'Jump': 3496, 'Walk': 3059, 'Stand': 2808, 'Throw': 1971, 'Sit': 1798, 'BodyContract': 1660, 'NoHuman': 1176, 'Fall': 1105, 'BodyRoll': 1058, 'Diving': 887, 'StandUp': 810, 'BasketballShot': 802, 'BasketballDunk': 791, 'Squat': 780, 'ClapHands': 680, 'TalkToCamera': 540, 'PoleVault': 519, 'PoleVaultPlantPole': 486, 'TwoHandedCatch': 464, 'CloseUpTalkToCamera': 444, 'HammerThrow': 441, 'HammerThrowRelease': 428, 'BodyBend': 411, 'HighJump': 406, 'HammerThrowSpin': 404, 'JavelinThrow': 361, 'CliffDiving': 360, 'TwoRaisedArmCelebrate': 353, 'CricketShot': 351, 'HammerThrowWindUp': 340, 'PickUp': 326, 'CricketBowling': 316, 'LongJump': 305, 'BasketballGuard': 298, 'BasketballDribble': 287, 'OneHandedCatch': 272, 'VolleyballSpiking': 266, 'DiscusWindUp': 229, 'OneRaisedArmCelebrate': 220, 'DiscusRelease': 218, 'Shotput': 214, 'BasketballPass': 212, 'TennisSwing': 210, 'ThrowDiscus': 208, 'Drop': 195, 'Billiards': 187, 'FistPump': 168, 'ShotPutBend': 152, 'FrisbeeCatch': 151, 'CleanAndJerk': 140, 'BasketballBlock': 134, 'WeightliftingJerk': 121, 'VolleyballSet': 118, 'BodyTurn': 117, 'SoccerPenalty': 114, 'WeightliftingClean': 110, 'VolleyballBlock': 101, 'Hug': 96, 'HighFive': 94, 'BaseballPitch': 71, 'GolfSwing': 67, 'PatPerson': 46, 'VolleyballServe': 30, 'VolleyballBump': 15}),
  index_to_token=('Run', 'Jump', 'Walk', 'Stand', 'Throw', 'Sit', 'BodyContract', 'NoHuman', 'Fall', 'BodyRoll', 'Diving', 'StandUp', 'BasketballShot', 'BasketballDunk', 'Squat', 'ClapHands', 'TalkToCamera', 'PoleVault', 'PoleVaultPlantPole', 'TwoHandedCatch', 'CloseUpTalkToCamera', 'HammerThrow', 'HammerThrowRelease', 'BodyBend', 'HighJump', 'HammerThrowSpin', 'JavelinThrow', 'CliffDiving', 'TwoRaisedArmCelebrate', 'CricketShot', 'HammerThrowWindUp', 'PickUp', 'CricketBowling', 'LongJump', 'BasketballGuard', 'BasketballDribble', 'OneHandedCatch', 'VolleyballSpiking', 'DiscusWindUp', 'OneRaisedArmCelebrate', 'DiscusRelease', 'Shotput', 'BasketballPass', 'TennisSwing', 'ThrowDiscus', 'Drop', 'Billiards', 'FistPump', 'ShotPutBend', 'FrisbeeCatch', 'CleanAndJerk', 'BasketballBlock', 'WeightliftingJerk', 'VolleyballSet', 'BodyTurn', 'SoccerPenalty', 'WeightliftingClean', 'VolleyballBlock', 'Hug', 'HighFive', 'BaseballPitch', 'GolfSwing', 'PatPerson', 'VolleyballServe', 'VolleyballBump'),
  token_to_index={'Run': 0, 'Jump': 1, 'Walk': 2, 'Stand': 3, 'Throw': 4, 'Sit': 5, 'BodyContract': 6, 'NoHuman': 7, 'Fall': 8, 'BodyRoll': 9, 'Diving': 10, 'StandUp': 11, 'BasketballShot': 12, 'BasketballDunk': 13, 'Squat': 14, 'ClapHands': 15, 'TalkToCamera': 16, 'PoleVault': 17, 'PoleVaultPlantPole': 18, 'TwoHandedCatch': 19, 'CloseUpTalkToCamera': 20, 'HammerThrow': 21, 'HammerThrowRelease': 22, 'BodyBend': 23, 'HighJump': 24, 'HammerThrowSpin': 25, 'JavelinThrow': 26, 'CliffDiving': 27, 'TwoRaisedArmCelebrate': 28, 'CricketShot': 29, 'HammerThrowWindUp': 30, 'PickUp': 31, 'CricketBowling': 32, 'LongJump': 33, 'BasketballGuard': 34, 'BasketballDribble': 35, 'OneHandedCatch': 36, 'VolleyballSpiking': 37, 'DiscusWindUp': 38, 'OneRaisedArmCelebrate': 39, 'DiscusRelease': 40, 'Shotput': 41, 'BasketballPass': 42, 'TennisSwing': 43, 'ThrowDiscus': 44, 'Drop': 45, 'Billiards': 46, 'FistPump': 47, 'ShotPutBend': 48, 'FrisbeeCatch': 49, 'CleanAndJerk': 50, 'BasketballBlock': 51, 'WeightliftingJerk': 52, 'VolleyballSet': 53, 'BodyTurn': 54, 'SoccerPenalty': 55, 'WeightliftingClean': 56, 'VolleyballBlock': 57, 'Hug': 58, 'HighFive': 59, 'BaseballPitch': 60, 'GolfSwing': 61, 'PatPerson': 62, 'VolleyballServe': 63, 'VolleyballBump': 64},
)}
```

It is possible to specify the dataset split to filter the data and keep only the data related to a
given dataset split.
The dataset has different can be decomposed in different splits. For example, the following line
will filter the data to keep only the rows related to the first training dataset (a.k.a. `train1`):

```pycon

>>> data, metadata = prepare_data(data_raw, split="validation")  # doctest: +SKIP

```

```textmate
shape: (18_482, 6)
┌─────────────────────────┬────────────┬────────────┬─────────────────────┬───────────┬────────────┐
│ video                   ┆ start_time ┆ end_time   ┆ action              ┆ action_id ┆ split      │
│ ---                     ┆ ---        ┆ ---        ┆ ---                 ┆ ---       ┆ ---        │
│ str                     ┆ f64        ┆ f64        ┆ str                 ┆ i64       ┆ str        │
╞═════════════════════════╪════════════╪════════════╪═════════════════════╪═══════════╪════════════╡
│ video_validation_000005 ┆ 0.03       ┆ 31.0       ┆ Stand               ┆ 3         ┆ validation │
│ 1                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000005 ┆ 0.03       ┆ 0.17       ┆ NoHuman             ┆ 7         ┆ validation │
│ 1                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000005 ┆ 0.2        ┆ 19.43      ┆ TalkToCamera        ┆ 16        ┆ validation │
│ 1                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000005 ┆ 0.2        ┆ 19.43      ┆ CloseUpTalkToCamera ┆ 20        ┆ validation │
│ 1                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000005 ┆ 37.299999  ┆ 47.93      ┆ NoHuman             ┆ 7         ┆ validation │
│ 1                       ┆            ┆            ┆                     ┆           ┆            │
│ …                       ┆ …          ┆ …          ┆ …                   ┆ …         ┆ …          │
│ video_validation_000099 ┆ 103.599998 ┆ 110.400002 ┆ VolleyballSpiking   ┆ 37        ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000099 ┆ 108.07     ┆ 111.57     ┆ Jump                ┆ 1         ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000099 ┆ 109.029999 ┆ 110.870003 ┆ BodyBend            ┆ 23        ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000099 ┆ 112.900002 ┆ 122.470001 ┆ CloseUpTalkToCamera ┆ 20        ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
│ video_validation_000099 ┆ 112.900002 ┆ 122.470001 ┆ TalkToCamera        ┆ 16        ┆ validation │
│ 0                       ┆            ┆            ┆                     ┆           ┆            │
└─────────────────────────┴────────────┴────────────┴─────────────────────┴───────────┴────────────┘
```

`arctix` provides the function `arctix.dataset.multithumos.to_array` to convert
the `polars.DataFrame` to a dictionary of numpy arrays.

```pycon

>>> from arctix.dataset.multithumos import to_array
>>> arrays = to_array(data)  # doctest: +SKIP

```

The dictionary contains some regular arrays and masked arrays because sequences have variable
lengths:

```textmate
{'sequence_length': array([ 25,  23,  32, ...,  48,  23,  44]),
 'split': array(['validation', 'validation', 'validation', ... 'validation', 'validation', 'validation'], dtype='<U10'),
 'action_id': masked_array(
   data=[[3, 7, 20, ..., --, --, --],
         [7, 7, 3, ..., --, --, --],
         [7, 46, 7, ..., --, --, --],
         ...,
         [7, 53, 37, ..., --, --, --],
         [7, 3, 4, ..., --, --, --],
         [16, 3, 63, ..., --, --, --]],
   mask=[[False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         ...,
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True]],
   fill_value=999999),
 'start_time': masked_array(
   data=[[0.029999999329447746, 0.029999999329447746, 0.20000000298023224,
          ..., --, --, --],
         [0.029999999329447746, 13.970000267028809, 22.170000076293945,
          ..., --, --, --],
         [0.029999999329447746, 9.100000381469727, 12.800000190734863,
          ..., --, --, --],
         ...,
         [0.029999999329447746, 7.070000171661377, 7.199999809265137, ...,
          --, --, --],
         [0.029999999329447746, 8.0, 8.0, ..., --, --, --],
         [0.029999999329447746, 0.029999999329447746, 51.0, ..., --, --,
          --]],
   mask=[[False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         ...,
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True]],
   fill_value=1e+20),
 'end_time': masked_array(
   data=[[31.0, 0.17000000178813934, 19.43000030517578, ..., --, --, --],
         [13.600000381469727, 23.3700008392334, 23.0, ..., --, --, --],
         [8.100000381469727, 13.800000190734863, 20.200000762939453, ...,
          --, --, --],
         ...,
         [7.03000020980835, 8.100000381469727, 10.199999809265137, ...,
          --, --, --],
         [8.0, 12.0, 10.0, ..., --, --, --],
         [49.06999969482422, 112.0, 53.0, ..., --, --, --]],
   mask=[[False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         ...,
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True]],
   fill_value=1e+20)}
```
