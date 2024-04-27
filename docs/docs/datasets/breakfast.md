# Breakfast

This page contains information about the Breakfast dataset, which was published with the following
paper:

```textmate
The Language of Actions: Recovering the Syntax and Semantics of Goal-
Directed Human Activities. Kuehne, Arslan, and Serre. CVPR 2014.
```

The data can be downloaded from
the [official project page](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/#Downloads).

## Get the raw data

`arctix` provides the function `arctix.dataset.breakfast.fetch_data` to easily load the raw data in
a `polars.DataFrame` format.

```pycon

>>> from pathlib import Path
>>> from arctix.dataset.breakfast import fetch_data
>>> dataset_path = Path("/path/to/dataset/breakfast")
>>> data_raw = fetch_data(dataset_path, name="segmentation_coarse")  # doctest: +SKIP
shape: (3_585, 5)
┌──────────────┬────────────┬──────────┬────────┬──────────────────┐
│ action       ┆ start_time ┆ end_time ┆ person ┆ cooking_activity │
│ ---          ┆ ---        ┆ ---      ┆ ---    ┆ ---              │
│ str          ┆ f64        ┆ f64      ┆ str    ┆ str              │
╞══════════════╪════════════╪══════════╪════════╪══════════════════╡
│ SIL          ┆ 1.0        ┆ 30.0     ┆ P03    ┆ cereals          │
│ take_bowl    ┆ 31.0       ┆ 150.0    ┆ P03    ┆ cereals          │
│ pour_cereals ┆ 151.0      ┆ 428.0    ┆ P03    ┆ cereals          │
│ pour_milk    ┆ 429.0      ┆ 575.0    ┆ P03    ┆ cereals          │
│ stir_cereals ┆ 576.0      ┆ 705.0    ┆ P03    ┆ cereals          │
│ …            ┆ …          ┆ …        ┆ …      ┆ …                │
│ take_cup     ┆ 38.0       ┆ 92.0     ┆ P54    ┆ tea              │
│ pour_water   ┆ 93.0       ┆ 229.0    ┆ P54    ┆ tea              │
│ add_teabag   ┆ 230.0      ┆ 744.0    ┆ P54    ┆ tea              │
│ pour_sugar   ┆ 745.0      ┆ 884.0    ┆ P54    ┆ tea              │
│ SIL          ┆ 885.0      ┆ 973.0    ┆ P54    ┆ tea              │
└──────────────┴────────────┴──────────┴────────┴──────────────────┘

```

There are two versions of the dataset: `"segmentation_coarse"` and `"segmentation_fine"`.
If the data is not downloaded in the dataset path, `fetch_data` automatically downloads the data.
You can set `force_download=True` to force to re-download the data if the data is already
downloaded.

## Prepare the data

`arctix` provides the function `arctix.dataset.breakfast.prepare_data` to preprocess the raw data.
It returns two outputs: the prepared data and the generate metadata.

```pycon

>>> from arctix.dataset.breakfast import prepare_data
>>> data, metadata = prepare_data(data_raw)  # doctest: +SKIP

```

`data` is a `polars.DataFrame` which contains the prepared data:

```textmate
shape: (3_585, 8)
┌─────────────┬────────────┬──────────┬────────┬─────────────┬───────────┬───────────┬─────────────┐
│ action      ┆ start_time ┆ end_time ┆ person ┆ cooking_act ┆ action_id ┆ person_id ┆ cooking_act │
│ ---         ┆ ---        ┆ ---      ┆ ---    ┆ ivity       ┆ ---       ┆ ---       ┆ ivity_id    │
│ str         ┆ f32        ┆ f32      ┆ str    ┆ ---         ┆ i64       ┆ i64       ┆ ---         │
│             ┆            ┆          ┆        ┆ str         ┆           ┆           ┆ i64         │
╞═════════════╪════════════╪══════════╪════════╪═════════════╪═══════════╪═══════════╪═════════════╡
│ SIL         ┆ 1.0        ┆ 30.0     ┆ P03    ┆ cereals     ┆ 0         ┆ 50        ┆ 7           │
│ take_bowl   ┆ 31.0       ┆ 150.0    ┆ P03    ┆ cereals     ┆ 16        ┆ 50        ┆ 7           │
│ pour_cereal ┆ 151.0      ┆ 428.0    ┆ P03    ┆ cereals     ┆ 21        ┆ 50        ┆ 7           │
│ s           ┆            ┆          ┆        ┆             ┆           ┆           ┆             │
│ pour_milk   ┆ 429.0      ┆ 575.0    ┆ P03    ┆ cereals     ┆ 1         ┆ 50        ┆ 7           │
│ stir_cereal ┆ 576.0      ┆ 705.0    ┆ P03    ┆ cereals     ┆ 37        ┆ 50        ┆ 7           │
│ s           ┆            ┆          ┆        ┆             ┆           ┆           ┆             │
│ …           ┆ …          ┆ …        ┆ …      ┆ …           ┆ …         ┆ …         ┆ …           │
│ take_cup    ┆ 38.0       ┆ 92.0     ┆ P54    ┆ tea         ┆ 13        ┆ 12        ┆ 8           │
│ pour_water  ┆ 93.0       ┆ 229.0    ┆ P54    ┆ tea         ┆ 23        ┆ 12        ┆ 8           │
│ add_teabag  ┆ 230.0      ┆ 744.0    ┆ P54    ┆ tea         ┆ 25        ┆ 12        ┆ 8           │
│ pour_sugar  ┆ 745.0      ┆ 884.0    ┆ P54    ┆ tea         ┆ 43        ┆ 12        ┆ 8           │
│ SIL         ┆ 885.0      ┆ 973.0    ┆ P54    ┆ tea         ┆ 0         ┆ 12        ┆ 8           │
└─────────────┴────────────┴──────────┴────────┴─────────────┴───────────┴───────────┴─────────────┘
```

You can note that new columns are added and the data types of some columns have changed.
`metadata` contains three vocabularies that are used to generate the new columns:

```textmate
{'vocab_action': Vocabulary(
   counter=Counter({'SIL': 1012, 'pour_milk': 199, 'cut_fruit': 175, 'crack_egg': 154, 'put_fruit2bowl': 139, 'take_plate': 132, 'put_egg2plate': 97, 'add_saltnpepper': 83, 'stir_dough': 78, 'peel_fruit': 74, 'fry_egg': 73, 'stirfry_egg': 67, 'butter_pan': 64, 'take_cup': 59, 'squeeze_orange': 59, 'pour_juice': 57, 'take_bowl': 56, 'put_toppingOnTop': 55, 'pour_oil': 55, 'fry_pancake': 55, 'spoon_powder': 53, 'pour_cereals': 53, 'stir_milk': 51, 'pour_water': 51, 'cut_bun': 51, 'add_teabag': 51, 'smear_butter': 49, 'pour_dough2pan': 49, 'pour_coffee': 49, 'spoon_flour': 48, 'cut_orange': 46, 'put_pancake2plate': 41, 'take_glass': 37, 'stir_egg': 30, 'take_knife': 28, 'put_bunTogether': 25, 'pour_egg2pan': 24, 'stir_cereals': 20, 'take_topping': 16, 'take_squeezer': 12, 'stir_fruit': 11, 'stir_coffee': 10, 'spoon_sugar': 10, 'pour_sugar': 8, 'pour_flour': 7, 'take_eggs': 6, 'take_butter': 4, 'stir_tea': 2}),
   index_to_token=('SIL', 'pour_milk', 'cut_fruit', 'crack_egg', 'put_fruit2bowl', 'take_plate', 'put_egg2plate', 'add_saltnpepper', 'stir_dough', 'peel_fruit', 'fry_egg', 'stirfry_egg', 'butter_pan', 'take_cup', 'squeeze_orange', 'pour_juice', 'take_bowl', 'put_toppingOnTop', 'pour_oil', 'fry_pancake', 'spoon_powder', 'pour_cereals', 'stir_milk', 'pour_water', 'cut_bun', 'add_teabag', 'smear_butter', 'pour_dough2pan', 'pour_coffee', 'spoon_flour', 'cut_orange', 'put_pancake2plate', 'take_glass', 'stir_egg', 'take_knife', 'put_bunTogether', 'pour_egg2pan', 'stir_cereals', 'take_topping', 'take_squeezer', 'stir_fruit', 'stir_coffee', 'spoon_sugar', 'pour_sugar', 'pour_flour', 'take_eggs', 'take_butter', 'stir_tea'),
   token_to_index={'SIL': 0, 'pour_milk': 1, 'cut_fruit': 2, 'crack_egg': 3, 'put_fruit2bowl': 4, 'take_plate': 5, 'put_egg2plate': 6, 'add_saltnpepper': 7, 'stir_dough': 8, 'peel_fruit': 9, 'fry_egg': 10, 'stirfry_egg': 11, 'butter_pan': 12, 'take_cup': 13, 'squeeze_orange': 14, 'pour_juice': 15, 'take_bowl': 16, 'put_toppingOnTop': 17, 'pour_oil': 18, 'fry_pancake': 19, 'spoon_powder': 20, 'pour_cereals': 21, 'stir_milk': 22, 'pour_water': 23, 'cut_bun': 24, 'add_teabag': 25, 'smear_butter': 26, 'pour_dough2pan': 27, 'pour_coffee': 28, 'spoon_flour': 29, 'cut_orange': 30, 'put_pancake2plate': 31, 'take_glass': 32, 'stir_egg': 33, 'take_knife': 34, 'put_bunTogether': 35, 'pour_egg2pan': 36, 'stir_cereals': 37, 'take_topping': 38, 'take_squeezer': 39, 'stir_fruit': 40, 'stir_coffee': 41, 'spoon_sugar': 42, 'pour_sugar': 43, 'pour_flour': 44, 'take_eggs': 45, 'take_butter': 46, 'stir_tea': 47},
 ),
 'vocab_activity': Vocabulary(
   counter=Counter({'pancake': 557, 'salat': 549, 'scrambledegg': 448, 'friedegg': 389, 'sandwich': 325, 'juice': 324, 'milk': 278, 'cereals': 256, 'tea': 233, 'coffee': 226}),
   index_to_token=('pancake', 'salat', 'scrambledegg', 'friedegg', 'sandwich', 'juice', 'milk', 'cereals', 'tea', 'coffee'),
   token_to_index={'pancake': 0, 'salat': 1, 'scrambledegg': 2, 'friedegg': 3, 'sandwich': 4, 'juice': 5, 'milk': 6, 'cereals': 7, 'tea': 8, 'coffee': 9},
 ),
 'vocab_person': Vocabulary(
   counter=Counter({'P47': 84, 'P31': 83, 'P05': 83, 'P16': 82, 'P51': 81, 'P09': 81, 'P07': 80, 'P50': 77, 'P41': 76, 'P27': 76, 'P20': 76, 'P18': 76, 'P54': 75, 'P32': 75, 'P43': 74, 'P38': 73, 'P30': 73, 'P23': 73, 'P08': 73, 'P39': 72, 'P53': 71, 'P49': 71, 'P42': 70, 'P22': 70, 'P48': 69, 'P14': 69, 'P10': 69, 'P06': 69, 'P35': 68, 'P25': 68, 'P04': 68, 'P44': 67, 'P33': 67, 'P21': 67, 'P17': 67, 'P52': 66, 'P11': 66, 'P46': 65, 'P40': 65, 'P24': 65, 'P19': 65, 'P13': 64, 'P45': 63, 'P26': 62, 'P37': 61, 'P12': 61, 'P15': 60, 'P36': 59, 'P29': 59, 'P34': 56, 'P03': 50, 'P28': 25}),
   index_to_token=('P47', 'P31', 'P05', 'P16', 'P51', 'P09', 'P07', 'P50', 'P41', 'P27', 'P20', 'P18', 'P54', 'P32', 'P43', 'P38', 'P30', 'P23', 'P08', 'P39', 'P53', 'P49', 'P42', 'P22', 'P48', 'P14', 'P10', 'P06', 'P35', 'P25', 'P04', 'P44', 'P33', 'P21', 'P17', 'P52', 'P11', 'P46', 'P40', 'P24', 'P19', 'P13', 'P45', 'P26', 'P37', 'P12', 'P15', 'P36', 'P29', 'P34', 'P03', 'P28'),
   token_to_index={'P47': 0, 'P31': 1, 'P05': 2, 'P16': 3, 'P51': 4, 'P09': 5, 'P07': 6, 'P50': 7, 'P41': 8, 'P27': 9, 'P20': 10, 'P18': 11, 'P54': 12, 'P32': 13, 'P43': 14, 'P38': 15, 'P30': 16, 'P23': 17, 'P08': 18, 'P39': 19, 'P53': 20, 'P49': 21, 'P42': 22, 'P22': 23, 'P48': 24, 'P14': 25, 'P10': 26, 'P06': 27, 'P35': 28, 'P25': 29, 'P04': 30, 'P44': 31, 'P33': 32, 'P21': 33, 'P17': 34, 'P52': 35, 'P11': 36, 'P46': 37, 'P40': 38, 'P24': 39, 'P19': 40, 'P13': 41, 'P45': 42, 'P26': 43, 'P37': 44, 'P12': 45, 'P15': 46, 'P36': 47, 'P29': 48, 'P34': 49, 'P03': 50, 'P28': 51},
 )}
```

It is possible to specify the dataset split to filter the data and keep only the data related to a
given dataset split.
The dataset has different can be decomposed in different splits. For example, the following line
will filter the data to keep only the rows related to the first training dataset (a.k.a. `train1`):

```pycon

>>> data, metadata = prepare_data(data_raw, split='train1')  # doctest: +SKIP

```

```textmate
shape: (2_692, 8)
┌─────────────┬────────────┬──────────┬────────┬─────────────┬───────────┬───────────┬─────────────┐
│ action      ┆ start_time ┆ end_time ┆ person ┆ cooking_act ┆ action_id ┆ person_id ┆ cooking_act │
│ ---         ┆ ---        ┆ ---      ┆ ---    ┆ ivity       ┆ ---       ┆ ---       ┆ ivity_id    │
│ str         ┆ f32        ┆ f32      ┆ str    ┆ ---         ┆ i64       ┆ i64       ┆ ---         │
│             ┆            ┆          ┆        ┆ str         ┆           ┆           ┆ i64         │
╞═════════════╪════════════╪══════════╪════════╪═════════════╪═══════════╪═══════════╪═════════════╡
│ SIL         ┆ 1.0        ┆ 9.0      ┆ P16    ┆ cereals     ┆ 0         ┆ 3         ┆ 7           │
│ pour_cereal ┆ 10.0       ┆ 269.0    ┆ P16    ┆ cereals     ┆ 21        ┆ 3         ┆ 7           │
│ s           ┆            ┆          ┆        ┆             ┆           ┆           ┆             │
│ pour_milk   ┆ 270.0      ┆ 474.0    ┆ P16    ┆ cereals     ┆ 1         ┆ 3         ┆ 7           │
│ SIL         ┆ 475.0      ┆ 548.0    ┆ P16    ┆ cereals     ┆ 0         ┆ 3         ┆ 7           │
│ SIL         ┆ 1.0        ┆ 39.0     ┆ P17    ┆ cereals     ┆ 0         ┆ 34        ┆ 7           │
│ …           ┆ …          ┆ …        ┆ …      ┆ …           ┆ …         ┆ …         ┆ …           │
│ take_cup    ┆ 38.0       ┆ 92.0     ┆ P54    ┆ tea         ┆ 13        ┆ 12        ┆ 8           │
│ pour_water  ┆ 93.0       ┆ 229.0    ┆ P54    ┆ tea         ┆ 23        ┆ 12        ┆ 8           │
│ add_teabag  ┆ 230.0      ┆ 744.0    ┆ P54    ┆ tea         ┆ 25        ┆ 12        ┆ 8           │
│ pour_sugar  ┆ 745.0      ┆ 884.0    ┆ P54    ┆ tea         ┆ 43        ┆ 12        ┆ 8           │
│ SIL         ┆ 885.0      ┆ 973.0    ┆ P54    ┆ tea         ┆ 0         ┆ 12        ┆ 8           │
└─────────────┴────────────┴──────────┴────────┴─────────────┴───────────┴───────────┴─────────────┘
```

`arctix` provides the function `arctix.dataset.breakfast.to_array` to convert
the `polars.DataFrame` to a dictionary of numpy arrays.

```pycon

>>> from arctix.dataset.breakfast import prepare_data
>>> arrays = to_array(data)  # doctest: +SKIP

```

The dictionary contains some regular arrays and masked arrays because sequences have variable
lengths:

```textmate
{'sequence_length': array([11, 19, 10,  ...,  5,  6,  4]),
 'person_id': array([ 0,  0,  0,  ..., 51, 51, 51]),
 'cooking_activity_id': array([0, 1, 2, ..., 4, 5, 7]),
 'action_id': masked_array(
   data=[[0, 3, 29, ..., --, --, --],
         [0, 5, 16, ..., --, --, --],
         [0, 3, 7, ..., --, --, --],
         ...,
         [0, 24, 26, ..., --, --, --],
         [0, 32, 30, ..., --, --, --],
         [0, 21, 1, ..., --, --, --]],
   mask=[[False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         ...,
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True]],
   fill_value=999999),
 'start_time': masked_array(
   data=[[1.0, 2.0, 441.0, ..., --, --, --],
         [1.0, 112.0, 227.0, ..., --, --, --],
         [1.0, 5.0, 397.0, ..., --, --, --],
         ...,
         [1.0, 385.0, 650.0, ..., --, --, --],
         [1.0, 31.0, 141.0, ..., --, --, --],
         [1.0, 45.0, 213.0, ..., --, --, --]],
   mask=[[False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         ...,
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True]],
   fill_value=1e+20),
 'end_time': masked_array(
   data=[[1.0, 440.0, 951.0, ..., --, --, --],
         [111.0, 226.0, 281.0, ..., --, --, --],
         [4.0, 396.0, 587.0, ..., --, --, --],
         ...,
         [384.0, 649.0, 1899.0, ..., --, --, --],
         [30.0, 140.0, 530.0, ..., --, --, --],
         [44.0, 212.0, 344.0, ..., --, --, --]],
   mask=[[False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         ...,
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True]],
   fill_value=1e+20)}
```
