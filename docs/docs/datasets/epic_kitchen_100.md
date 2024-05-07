# EPIC-KITCHENS-100

This page contains information about the EPIC-KITCHENS-100 dataset, which was published with the
following paper:

```textmate
Rescaling Egocentric Vision
Damen D., Doughty H., Farinella G., Furnari A., Ma J., Kazakos E., Moltisanti D.,
Munro J., Perrett T., Price W., and Wray M.
IJCV 2022 (https://arxiv.org/abs/2006.13256)
```

The data can be downloaded from
the [official project page](https://epic-kitchens.github.io/2024) and the
associated [GitHub repository](https://github.com/epic-kitchens/epic-kitchens-100-annotations).

## Get the raw data

`arctix` provides the function `arctix.dataset.epic_kitchen_100.fetch_data` to easily load the raw
data in a `polars.DataFrame` format.

```pycon

>>> from pathlib import Path
>>> from arctix.dataset.epic_kitchen_100 import fetch_data
>>> dataset_path = Path("/path/to/dataset/epic_kitchen_100")
>>> data_raw, metadata_raw = fetch_data(dataset_path, split="train")  # doctest: +SKIP
shape: (67_217, 15)
┌────────────┬───────────┬───────────┬───────────┬───┬───────────┬──────────┬───────────┬──────────┐
│ all_noun_c ┆ all_nouns ┆ narration ┆ narration ┆ … ┆ stop_time ┆ verb     ┆ verb_clas ┆ video_id │
│ lasses     ┆ ---       ┆ ---       ┆ _id       ┆   ┆ stamp     ┆ ---      ┆ s         ┆ ---      │
│ ---        ┆ list[str] ┆ str       ┆ ---       ┆   ┆ ---       ┆ str      ┆ ---       ┆ str      │
│ list[i64]  ┆           ┆           ┆ str       ┆   ┆ time      ┆          ┆ i64       ┆          │
╞════════════╪═══════════╪═══════════╪═══════════╪═══╪═══════════╪══════════╪═══════════╪══════════╡
│ [3]        ┆ ["door"]  ┆ open door ┆ P01_01_0  ┆ … ┆ 00:00:03. ┆ open     ┆ 3         ┆ P01_01   │
│            ┆           ┆           ┆           ┆   ┆ 370       ┆          ┆           ┆          │
│ [114]      ┆ ["light"] ┆ turn on   ┆ P01_01_1  ┆ … ┆ 00:00:06. ┆ turn-on  ┆ 6         ┆ P01_01   │
│            ┆           ┆ light     ┆           ┆   ┆ 170       ┆          ┆           ┆          │
│ [3]        ┆ ["door"]  ┆ close     ┆ P01_01_2  ┆ … ┆ 00:00:09. ┆ close    ┆ 4         ┆ P01_01   │
│            ┆           ┆ door      ┆           ┆   ┆ 490       ┆          ┆           ┆          │
│ [12]       ┆ ["fridge" ┆ open      ┆ P01_01_3  ┆ … ┆ 00:00:13. ┆ open     ┆ 3         ┆ P01_01   │
│            ┆ ]         ┆ fridge    ┆           ┆   ┆ 990       ┆          ┆           ┆          │
│ [223]      ┆ ["celery" ┆ take      ┆ P01_01_4  ┆ … ┆ 00:00:16. ┆ take     ┆ 0         ┆ P01_01   │
│            ┆ ]         ┆ celery    ┆           ┆   ┆ 400       ┆          ┆           ┆          │
│ …          ┆ …         ┆ …         ┆ …         ┆ … ┆ …         ┆ …        ┆ …         ┆ …        │
│ [0]        ┆ ["tap"]   ┆ turn on   ┆ P37_103_6 ┆ … ┆ 00:06:16. ┆ turn-on  ┆ 6         ┆ P37_103  │
│            ┆           ┆ tap       ┆ 9         ┆   ┆ 690       ┆          ┆           ┆          │
│ [11]       ┆ ["hand"]  ┆ wash      ┆ P37_103_7 ┆ … ┆ 00:06:17. ┆ wash     ┆ 2         ┆ P37_103  │
│            ┆           ┆ hands     ┆ 0         ┆   ┆ 290       ┆          ┆           ┆          │
│ [0]        ┆ ["tap"]   ┆ turn off  ┆ P37_103_7 ┆ … ┆ 00:06:17. ┆ turn-off ┆ 8         ┆ P37_103  │
│            ┆           ┆ tap       ┆ 1         ┆   ┆ 670       ┆          ┆           ┆          │
│ [5]        ┆ ["pan"]   ┆ take pan  ┆ P37_103_7 ┆ … ┆ 00:06:23. ┆ take     ┆ 0         ┆ P37_103  │
│            ┆           ┆           ┆ 2         ┆   ┆ 770       ┆          ┆           ┆          │
│ [27]       ┆ ["water:b ┆ pour out  ┆ P37_103_7 ┆ … ┆ 00:06:32. ┆ pour-out ┆ 9         ┆ P37_103  │
│            ┆ oiled"]   ┆ boiled    ┆ 3         ┆   ┆ 660       ┆          ┆           ┆          │
│            ┆           ┆ water     ┆           ┆   ┆           ┆          ┆           ┆          │
└────────────┴───────────┴───────────┴───────────┴───┴───────────┴──────────┴───────────┴──────────┘

(vocab_noun): Vocabulary(vocab_size=300)
(vocab_verb): Vocabulary(vocab_size=97)

```

If the data is not downloaded in the dataset path, `fetch_data` automatically downloads the data.
You can set `force_download=True` to force to re-download the data if the data is already
downloaded.

## Prepare the data

`arctix` provides the function `arctix.dataset.epic_kitchen_100.prepare_data` to preprocess the raw
data.
It returns two outputs: the prepared data and the generate metadata.

```pycon

>>> from arctix.dataset.epic_kitchen_100 import prepare_data
>>> data, metadata = prepare_data(data_raw, metadata_raw)  # doctest: +SKIP

```

`data` is a `polars.DataFrame` which contains the prepared data:

```textmate
shape: (67_217, 17)
┌────────────┬───────────┬───────────┬───────────┬───┬───────────┬──────────┬───────────┬──────────┐
│ all_noun_c ┆ all_nouns ┆ narration ┆ narration ┆ … ┆ stop_time ┆ verb     ┆ verb_clas ┆ video_id │
│ lasses     ┆ ---       ┆ ---       ┆ _id       ┆   ┆ stamp     ┆ ---      ┆ s         ┆ ---      │
│ ---        ┆ list[str] ┆ str       ┆ ---       ┆   ┆ ---       ┆ str      ┆ ---       ┆ str      │
│ list[i64]  ┆           ┆           ┆ str       ┆   ┆ time      ┆          ┆ i64       ┆          │
╞════════════╪═══════════╪═══════════╪═══════════╪═══╪═══════════╪══════════╪═══════════╪══════════╡
│ [3]        ┆ ["door"]  ┆ open door ┆ P01_01_0  ┆ … ┆ 00:00:03. ┆ open     ┆ 3         ┆ P01_01   │
│            ┆           ┆           ┆           ┆   ┆ 370       ┆          ┆           ┆          │
│ [114]      ┆ ["light"] ┆ turn on   ┆ P01_01_1  ┆ … ┆ 00:00:06. ┆ turn-on  ┆ 6         ┆ P01_01   │
│            ┆           ┆ light     ┆           ┆   ┆ 170       ┆          ┆           ┆          │
│ [3]        ┆ ["door"]  ┆ close     ┆ P01_01_2  ┆ … ┆ 00:00:09. ┆ close    ┆ 4         ┆ P01_01   │
│            ┆           ┆ door      ┆           ┆   ┆ 490       ┆          ┆           ┆          │
│ [12]       ┆ ["fridge" ┆ open      ┆ P01_01_3  ┆ … ┆ 00:00:13. ┆ open     ┆ 3         ┆ P01_01   │
│            ┆ ]         ┆ fridge    ┆           ┆   ┆ 990       ┆          ┆           ┆          │
│ [223]      ┆ ["celery" ┆ take      ┆ P01_01_4  ┆ … ┆ 00:00:16. ┆ take     ┆ 0         ┆ P01_01   │
│            ┆ ]         ┆ celery    ┆           ┆   ┆ 400       ┆          ┆           ┆          │
│ …          ┆ …         ┆ …         ┆ …         ┆ … ┆ …         ┆ …        ┆ …         ┆ …        │
│ [0]        ┆ ["tap"]   ┆ turn on   ┆ P37_103_6 ┆ … ┆ 00:06:16. ┆ turn-on  ┆ 6         ┆ P37_103  │
│            ┆           ┆ tap       ┆ 9         ┆   ┆ 690       ┆          ┆           ┆          │
│ [11]       ┆ ["hand"]  ┆ wash      ┆ P37_103_7 ┆ … ┆ 00:06:17. ┆ wash     ┆ 2         ┆ P37_103  │
│            ┆           ┆ hands     ┆ 0         ┆   ┆ 290       ┆          ┆           ┆          │
│ [0]        ┆ ["tap"]   ┆ turn off  ┆ P37_103_7 ┆ … ┆ 00:06:17. ┆ turn-off ┆ 8         ┆ P37_103  │
│            ┆           ┆ tap       ┆ 1         ┆   ┆ 670       ┆          ┆           ┆          │
│ [5]        ┆ ["pan"]   ┆ take pan  ┆ P37_103_7 ┆ … ┆ 00:06:23. ┆ take     ┆ 0         ┆ P37_103  │
│            ┆           ┆           ┆ 2         ┆   ┆ 770       ┆          ┆           ┆          │
│ [27]       ┆ ["water:b ┆ pour out  ┆ P37_103_7 ┆ … ┆ 00:06:32. ┆ pour-out ┆ 9         ┆ P37_103  │
│            ┆ oiled"]   ┆ boiled    ┆ 3         ┆   ┆ 660       ┆          ┆           ┆          │
│            ┆           ┆ water     ┆           ┆   ┆           ┆          ┆           ┆          │
└────────────┴───────────┴───────────┴───────────┴───┴───────────┴──────────┴───────────┴──────────┘
```

You can note that new columns are added and the data types of some columns have changed.
`metadata` contains a vocabulary that is used to generate the column `action_id`:

```textmate
{'vocab_noun': Vocabulary(
  counter=Counter({'tap': 1, 'spoon': 1, 'plate': 1, 'cupboard': 1, 'knife': 1, 'pan': 1, 'lid': 1, 'bowl': 1, 'drawer': 1, 'sponge': 1, 'glass': 1, 'hand': 1, 'fridge': 1, 'cup': 1, 'fork': 1, 'bottle': 1, 'onion': 1, 'cloth': 1, 'board:chopping': 1, 'bag': 1, 'spatula': 1, 'container': 1, 'liquid:washing': 1, 'box': 1, 'hob': 1, 'dough': 1, 'package': 1, 'water': 1, 'meat': 1, 'pot': 1, 'potato': 1, 'oil': 1, 'cheese': 1, 'bread': 1, 'food': 1, 'tray': 1, 'bin': 1, 'pepper': 1, 'salt': 1, 'colander': 1, 'jar': 1, 'carrot': 1, 'top': 1, 'tomato': 1, 'kettle': 1, 'pasta': 1, 'oven': 1, 'sauce': 1, 'skin': 1, 'paper': 1, 'maker:coffee': 1, 'garlic': 1, 'towel': 1, 'egg': 1, 'rubbish': 1, 'rice': 1, 'mushroom': 1, 'chicken': 1, 'cutlery': 1, 'coffee': 1, 'glove': 1, 'can': 1, 'leaf': 1, 'sink': 1, 'milk': 1, 'heat': 1, 'jug': 1, 'aubergine': 1, 'salad': 1, 'chilli': 1, 'dishwasher': 1, 'mixture': 1, 'cucumber': 1, 'clothes': 1, 'peach': 1, 'flour': 1, 'courgette': 1, 'filter': 1, 'butter': 1, 'scissors': 1, 'chopstick': 1, 'tofu': 1, 'blender': 1, 'olive': 1, 'mat': 1, 'spice': 1, 'sausage': 1, 'peeler:potato': 1, 'napkin': 1, 'cover': 1, 'microwave': 1, 'pizza': 1, 'button': 1, 'towel:kitchen': 1, 'vegetable': 1, 'stock': 1, 'grater': 1, 'ladle': 1, 'yoghurt': 1, 'cereal': 1, 'wrap:plastic': 1, 'broccoli': 1, 'sugar': 1, 'brush': 1, 'biscuit': 1, 'lemon': 1, 'juicer': 1, 'wrap': 1, 'scale': 1, 'rest': 1, 'rack:drying': 1, 'alarm': 1, 'salmon': 1, 'freezer': 1, 'light': 1, 'spreads': 1, 'squash': 1, 'leek': 1, 'cap': 1, 'fish': 1, 'lettuce': 1, 'curry': 1, 'seed': 1, 'foil': 1, 'machine:washing': 1, 'corn': 1, 'soup': 1, 'oatmeal': 1, 'onion:spring': 1, 'clip': 1, 'lighter': 1, 'ginger': 1, 'tea': 1, 'nut': 1, 'vinegar': 1, 'holder': 1, 'pin:rolling': 1, 'pie': 1, 'powder': 1, 'burger': 1, 'book': 1, 'shell:egg': 1, 'tongs': 1, 'cream': 1, 'pork': 1, 'oregano': 1, 'banana': 1, 'processor:food': 1, 'paste': 1, 'recipe': 1, 'liquid': 1, 'choi:pak': 1, 'cooker:slow': 1, 'plug': 1, 'utensil': 1, 'noodle': 1, 'salami': 1, 'kitchen': 1, 'teapot': 1, 'floor': 1, 'tuna': 1, 'lime': 1, 'omelette': 1, 'bacon': 1, 'sandwich': 1, 'phone': 1, 'thermometer': 1, 'orange': 1, 'basket': 1, 'parsley': 1, 'spinner:salad': 1, 'tablet': 1, 'presser': 1, 'coriander': 1, 'opener:bottle': 1, 'cake': 1, 'avocado': 1, 'lentil': 1, 'blueberry': 1, 'fan:extractor': 1, 'cellar:salt': 1, 'hummus': 1, 'chair': 1, 'juice': 1, 'pancake': 1, 'bean:green': 1, 'toaster': 1, 'apple': 1, 'chocolate': 1, 'ice': 1, 'knob': 1, 'handle': 1, 'wine': 1, 'pea': 1, 'pith': 1, 'yeast': 1, 'coconut': 1, 'fishcakes': 1, 'spinach': 1, 'apron': 1, 'raisin': 1, 'basil': 1, 'grape': 1, 'kale': 1, 'wire': 1, 'asparagus': 1, 'paprika': 1, 'mango': 1, 'caper': 1, 'drink': 1, 'stalk': 1, 'turmeric': 1, 'whetstone': 1, 'kiwi': 1, 'bean': 1, 'thyme': 1, 'finger:lady': 1, 'beef': 1, 'whisk': 1, 'blackberry': 1, 'slicer': 1, 'control:remote': 1, 'label': 1, 'celery': 1, 'cabbage': 1, 'hoover': 1, 'breadstick': 1, 'roll': 1, 'cocktail': 1, 'crisp': 1, 'ladder': 1, 'beer': 1, 'pan:dust': 1, 'battery': 1, 'powder:washing': 1, 'backpack': 1, 'cumin': 1, 'cutter:pizza': 1, 'air': 1, 'pear': 1, 'quorn': 1, 'funnel': 1, 'wall': 1, 'strawberry': 1, 'almond': 1, 'tv': 1, 'scotch:egg': 1, 'shelf': 1, 'straw': 1, 'stand': 1, 'machine:sous:vide': 1, 'masher': 1, 'guard:hand': 1, 'shrimp': 1, 'fruit': 1, 'artichoke': 1, 'cork': 1, 'cherry': 1, 'sprout': 1, 'mat:sushi': 1, 'stick:crab': 1, 'ring:onion': 1, 'pestle': 1, 'window': 1, 'gin': 1, 'bar': 1, 'mint': 1, 'heater': 1, 'grass:lemon': 1, 'rubber': 1, 'gherkin': 1, 'breadcrumb': 1, 'watch': 1, 'melon': 1, 'cinnamon': 1, 'popcorn': 1, 'dumpling': 1, 'rosemary': 1, 'power': 1, 'syrup': 1, 'candle': 1, 'pineapple': 1, 'sheets': 1, 'soda': 1, 'raspberry': 1, 'airer': 1, 'balloon': 1, 'turkey': 1, 'computer': 1, 'key': 1, 'pillow': 1, 'pen': 1, 'face': 1, 'plum': 1, 'whiskey': 1, 'door:kitchen': 1, 'tape': 1, 'camera': 1, 'cd': 1, 'extract:vanilla': 1}),
  index_to_token=('tap', 'spoon', 'plate', 'cupboard', 'knife', 'pan', 'lid', 'bowl', 'drawer', 'sponge', 'glass', 'hand', 'fridge', 'cup', 'fork', 'bottle', 'onion', 'cloth', 'board:chopping', 'bag', 'spatula', 'container', 'liquid:washing', 'box', 'hob', 'dough', 'package', 'water', 'meat', 'pot', 'potato', 'oil', 'cheese', 'bread', 'food', 'tray', 'bin', 'pepper', 'salt', 'colander', 'jar', 'carrot', 'top', 'tomato', 'kettle', 'pasta', 'oven', 'sauce', 'skin', 'paper', 'maker:coffee', 'garlic', 'towel', 'egg', 'rubbish', 'rice', 'mushroom', 'chicken', 'cutlery', 'coffee', 'glove', 'can', 'leaf', 'sink', 'milk', 'heat', 'jug', 'aubergine', 'salad', 'chilli', 'dishwasher', 'mixture', 'cucumber', 'clothes', 'peach', 'flour', 'courgette', 'filter', 'butter', 'scissors', 'chopstick', 'tofu', 'blender', 'olive', 'mat', 'spice', 'sausage', 'peeler:potato', 'napkin', 'cover', 'microwave', 'pizza', 'button', 'towel:kitchen', 'vegetable', 'stock', 'grater', 'ladle', 'yoghurt', 'cereal', 'wrap:plastic', 'broccoli', 'sugar', 'brush', 'biscuit', 'lemon', 'juicer', 'wrap', 'scale', 'rest', 'rack:drying', 'alarm', 'salmon', 'freezer', 'light', 'spreads', 'squash', 'leek', 'cap', 'fish', 'lettuce', 'curry', 'seed', 'foil', 'machine:washing', 'corn', 'soup', 'oatmeal', 'onion:spring', 'clip', 'lighter', 'ginger', 'tea', 'nut', 'vinegar', 'holder', 'pin:rolling', 'pie', 'powder', 'burger', 'book', 'shell:egg', 'tongs', 'cream', 'pork', 'oregano', 'banana', 'processor:food', 'paste', 'recipe', 'liquid', 'choi:pak', 'cooker:slow', 'plug', 'utensil', 'noodle', 'salami', 'kitchen', 'teapot', 'floor', 'tuna', 'lime', 'omelette', 'bacon', 'sandwich', 'phone', 'thermometer', 'orange', 'basket', 'parsley', 'spinner:salad', 'tablet', 'presser', 'coriander', 'opener:bottle', 'cake', 'avocado', 'lentil', 'blueberry', 'fan:extractor', 'cellar:salt', 'hummus', 'chair', 'juice', 'pancake', 'bean:green', 'toaster', 'apple', 'chocolate', 'ice', 'knob', 'handle', 'wine', 'pea', 'pith', 'yeast', 'coconut', 'fishcakes', 'spinach', 'apron', 'raisin', 'basil', 'grape', 'kale', 'wire', 'asparagus', 'paprika', 'mango', 'caper', 'drink', 'stalk', 'turmeric', 'whetstone', 'kiwi', 'bean', 'thyme', 'finger:lady', 'beef', 'whisk', 'blackberry', 'slicer', 'control:remote', 'label', 'celery', 'cabbage', 'hoover', 'breadstick', 'roll', 'cocktail', 'crisp', 'ladder', 'beer', 'pan:dust', 'battery', 'powder:washing', 'backpack', 'cumin', 'cutter:pizza', 'air', 'pear', 'quorn', 'funnel', 'wall', 'strawberry', 'almond', 'tv', 'scotch:egg', 'shelf', 'straw', 'stand', 'machine:sous:vide', 'masher', 'guard:hand', 'shrimp', 'fruit', 'artichoke', 'cork', 'cherry', 'sprout', 'mat:sushi', 'stick:crab', 'ring:onion', 'pestle', 'window', 'gin', 'bar', 'mint', 'heater', 'grass:lemon', 'rubber', 'gherkin', 'breadcrumb', 'watch', 'melon', 'cinnamon', 'popcorn', 'dumpling', 'rosemary', 'power', 'syrup', 'candle', 'pineapple', 'sheets', 'soda', 'raspberry', 'airer', 'balloon', 'turkey', 'computer', 'key', 'pillow', 'pen', 'face', 'plum', 'whiskey', 'door:kitchen', 'tape', 'camera', 'cd', 'extract:vanilla'),
  token_to_index={'tap': 0, 'spoon': 1, 'plate': 2, 'cupboard': 3, 'knife': 4, 'pan': 5, 'lid': 6, 'bowl': 7, 'drawer': 8, 'sponge': 9, 'glass': 10, 'hand': 11, 'fridge': 12, 'cup': 13, 'fork': 14, 'bottle': 15, 'onion': 16, 'cloth': 17, 'board:chopping': 18, 'bag': 19, 'spatula': 20, 'container': 21, 'liquid:washing': 22, 'box': 23, 'hob': 24, 'dough': 25, 'package': 26, 'water': 27, 'meat': 28, 'pot': 29, 'potato': 30, 'oil': 31, 'cheese': 32, 'bread': 33, 'food': 34, 'tray': 35, 'bin': 36, 'pepper': 37, 'salt': 38, 'colander': 39, 'jar': 40, 'carrot': 41, 'top': 42, 'tomato': 43, 'kettle': 44, 'pasta': 45, 'oven': 46, 'sauce': 47, 'skin': 48, 'paper': 49, 'maker:coffee': 50, 'garlic': 51, 'towel': 52, 'egg': 53, 'rubbish': 54, 'rice': 55, 'mushroom': 56, 'chicken': 57, 'cutlery': 58, 'coffee': 59, 'glove': 60, 'can': 61, 'leaf': 62, 'sink': 63, 'milk': 64, 'heat': 65, 'jug': 66, 'aubergine': 67, 'salad': 68, 'chilli': 69, 'dishwasher': 70, 'mixture': 71, 'cucumber': 72, 'clothes': 73, 'peach': 74, 'flour': 75, 'courgette': 76, 'filter': 77, 'butter': 78, 'scissors': 79, 'chopstick': 80, 'tofu': 81, 'blender': 82, 'olive': 83, 'mat': 84, 'spice': 85, 'sausage': 86, 'peeler:potato': 87, 'napkin': 88, 'cover': 89, 'microwave': 90, 'pizza': 91, 'button': 92, 'towel:kitchen': 93, 'vegetable': 94, 'stock': 95, 'grater': 96, 'ladle': 97, 'yoghurt': 98, 'cereal': 99, 'wrap:plastic': 100, 'broccoli': 101, 'sugar': 102, 'brush': 103, 'biscuit': 104, 'lemon': 105, 'juicer': 106, 'wrap': 107, 'scale': 108, 'rest': 109, 'rack:drying': 110, 'alarm': 111, 'salmon': 112, 'freezer': 113, 'light': 114, 'spreads': 115, 'squash': 116, 'leek': 117, 'cap': 118, 'fish': 119, 'lettuce': 120, 'curry': 121, 'seed': 122, 'foil': 123, 'machine:washing': 124, 'corn': 125, 'soup': 126, 'oatmeal': 127, 'onion:spring': 128, 'clip': 129, 'lighter': 130, 'ginger': 131, 'tea': 132, 'nut': 133, 'vinegar': 134, 'holder': 135, 'pin:rolling': 136, 'pie': 137, 'powder': 138, 'burger': 139, 'book': 140, 'shell:egg': 141, 'tongs': 142, 'cream': 143, 'pork': 144, 'oregano': 145, 'banana': 146, 'processor:food': 147, 'paste': 148, 'recipe': 149, 'liquid': 150, 'choi:pak': 151, 'cooker:slow': 152, 'plug': 153, 'utensil': 154, 'noodle': 155, 'salami': 156, 'kitchen': 157, 'teapot': 158, 'floor': 159, 'tuna': 160, 'lime': 161, 'omelette': 162, 'bacon': 163, 'sandwich': 164, 'phone': 165, 'thermometer': 166, 'orange': 167, 'basket': 168, 'parsley': 169, 'spinner:salad': 170, 'tablet': 171, 'presser': 172, 'coriander': 173, 'opener:bottle': 174, 'cake': 175, 'avocado': 176, 'lentil': 177, 'blueberry': 178, 'fan:extractor': 179, 'cellar:salt': 180, 'hummus': 181, 'chair': 182, 'juice': 183, 'pancake': 184, 'bean:green': 185, 'toaster': 186, 'apple': 187, 'chocolate': 188, 'ice': 189, 'knob': 190, 'handle': 191, 'wine': 192, 'pea': 193, 'pith': 194, 'yeast': 195, 'coconut': 196, 'fishcakes': 197, 'spinach': 198, 'apron': 199, 'raisin': 200, 'basil': 201, 'grape': 202, 'kale': 203, 'wire': 204, 'asparagus': 205, 'paprika': 206, 'mango': 207, 'caper': 208, 'drink': 209, 'stalk': 210, 'turmeric': 211, 'whetstone': 212, 'kiwi': 213, 'bean': 214, 'thyme': 215, 'finger:lady': 216, 'beef': 217, 'whisk': 218, 'blackberry': 219, 'slicer': 220, 'control:remote': 221, 'label': 222, 'celery': 223, 'cabbage': 224, 'hoover': 225, 'breadstick': 226, 'roll': 227, 'cocktail': 228, 'crisp': 229, 'ladder': 230, 'beer': 231, 'pan:dust': 232, 'battery': 233, 'powder:washing': 234, 'backpack': 235, 'cumin': 236, 'cutter:pizza': 237, 'air': 238, 'pear': 239, 'quorn': 240, 'funnel': 241, 'wall': 242, 'strawberry': 243, 'almond': 244, 'tv': 245, 'scotch:egg': 246, 'shelf': 247, 'straw': 248, 'stand': 249, 'machine:sous:vide': 250, 'masher': 251, 'guard:hand': 252, 'shrimp': 253, 'fruit': 254, 'artichoke': 255, 'cork': 256, 'cherry': 257, 'sprout': 258, 'mat:sushi': 259, 'stick:crab': 260, 'ring:onion': 261, 'pestle': 262, 'window': 263, 'gin': 264, 'bar': 265, 'mint': 266, 'heater': 267, 'grass:lemon': 268, 'rubber': 269, 'gherkin': 270, 'breadcrumb': 271, 'watch': 272, 'melon': 273, 'cinnamon': 274, 'popcorn': 275, 'dumpling': 276, 'rosemary': 277, 'power': 278, 'syrup': 279, 'candle': 280, 'pineapple': 281, 'sheets': 282, 'soda': 283, 'raspberry': 284, 'airer': 285, 'balloon': 286, 'turkey': 287, 'computer': 288, 'key': 289, 'pillow': 290, 'pen': 291, 'face': 292, 'plum': 293, 'whiskey': 294, 'door:kitchen': 295, 'tape': 296, 'camera': 297, 'cd': 298, 'extract:vanilla': 299},
),
'vocab_verb': Vocabulary(
  counter=Counter({'take': 1, 'put': 1, 'wash': 1, 'open': 1, 'close': 1, 'insert': 1, 'turn-on': 1, 'cut': 1, 'turn-off': 1, 'pour': 1, 'mix': 1, 'move': 1, 'remove': 1, 'throw': 1, 'dry': 1, 'shake': 1, 'scoop': 1, 'adjust': 1, 'squeeze': 1, 'peel': 1, 'empty': 1, 'press': 1, 'flip': 1, 'turn': 1, 'check': 1, 'scrape': 1, 'fill': 1, 'apply': 1, 'fold': 1, 'scrub': 1, 'break': 1, 'pull': 1, 'pat': 1, 'lift': 1, 'hold': 1, 'eat': 1, 'wrap': 1, 'filter': 1, 'look': 1, 'unroll': 1, 'sort': 1, 'hang': 1, 'sprinkle': 1, 'rip': 1, 'spray': 1, 'cook': 1, 'add': 1, 'roll': 1, 'search': 1, 'crush': 1, 'stretch': 1, 'knead': 1, 'divide': 1, 'set': 1, 'feel': 1, 'rub': 1, 'soak': 1, 'brush': 1, 'sharpen': 1, 'drop': 1, 'drink': 1, 'slide': 1, 'water': 1, 'gather': 1, 'attach': 1, 'turn-down': 1, 'coat': 1, 'transition': 1, 'wear': 1, 'measure': 1, 'increase': 1, 'unscrew': 1, 'wait': 1, 'lower': 1, 'form': 1, 'smell': 1, 'use': 1, 'grate': 1, 'screw': 1, 'let-go': 1, 'finish': 1, 'stab': 1, 'serve': 1, 'uncover': 1, 'unwrap': 1, 'choose': 1, 'lock': 1, 'flatten': 1, 'switch': 1, 'carry': 1, 'season': 1, 'unlock': 1, 'prepare': 1, 'bake': 1, 'mark': 1, 'bend': 1, 'unfreeze': 1}),
  index_to_token=('take', 'put', 'wash', 'open', 'close', 'insert', 'turn-on', 'cut', 'turn-off', 'pour', 'mix', 'move', 'remove', 'throw', 'dry', 'shake', 'scoop', 'adjust', 'squeeze', 'peel', 'empty', 'press', 'flip', 'turn', 'check', 'scrape', 'fill', 'apply', 'fold', 'scrub', 'break', 'pull', 'pat', 'lift', 'hold', 'eat', 'wrap', 'filter', 'look', 'unroll', 'sort', 'hang', 'sprinkle', 'rip', 'spray', 'cook', 'add', 'roll', 'search', 'crush', 'stretch', 'knead', 'divide', 'set', 'feel', 'rub', 'soak', 'brush', 'sharpen', 'drop', 'drink', 'slide', 'water', 'gather', 'attach', 'turn-down', 'coat', 'transition', 'wear', 'measure', 'increase', 'unscrew', 'wait', 'lower', 'form', 'smell', 'use', 'grate', 'screw', 'let-go', 'finish', 'stab', 'serve', 'uncover', 'unwrap', 'choose', 'lock', 'flatten', 'switch', 'carry', 'season', 'unlock', 'prepare', 'bake', 'mark', 'bend', 'unfreeze'),
  token_to_index={'take': 0, 'put': 1, 'wash': 2, 'open': 3, 'close': 4, 'insert': 5, 'turn-on': 6, 'cut': 7, 'turn-off': 8, 'pour': 9, 'mix': 10, 'move': 11, 'remove': 12, 'throw': 13, 'dry': 14, 'shake': 15, 'scoop': 16, 'adjust': 17, 'squeeze': 18, 'peel': 19, 'empty': 20, 'press': 21, 'flip': 22, 'turn': 23, 'check': 24, 'scrape': 25, 'fill': 26, 'apply': 27, 'fold': 28, 'scrub': 29, 'break': 30, 'pull': 31, 'pat': 32, 'lift': 33, 'hold': 34, 'eat': 35, 'wrap': 36, 'filter': 37, 'look': 38, 'unroll': 39, 'sort': 40, 'hang': 41, 'sprinkle': 42, 'rip': 43, 'spray': 44, 'cook': 45, 'add': 46, 'roll': 47, 'search': 48, 'crush': 49, 'stretch': 50, 'knead': 51, 'divide': 52, 'set': 53, 'feel': 54, 'rub': 55, 'soak': 56, 'brush': 57, 'sharpen': 58, 'drop': 59, 'drink': 60, 'slide': 61, 'water': 62, 'gather': 63, 'attach': 64, 'turn-down': 65, 'coat': 66, 'transition': 67, 'wear': 68, 'measure': 69, 'increase': 70, 'unscrew': 71, 'wait': 72, 'lower': 73, 'form': 74, 'smell': 75, 'use': 76, 'grate': 77, 'screw': 78, 'let-go': 79, 'finish': 80, 'stab': 81, 'serve': 82, 'uncover': 83, 'unwrap': 84, 'choose': 85, 'lock': 86, 'flatten': 87, 'switch': 88, 'carry': 89, 'season': 90, 'unlock': 91, 'prepare': 92, 'bake': 93, 'mark': 94, 'bend': 95, 'unfreeze': 96},
)}
```

`arctix` provides the function `arctix.dataset.epic_kitchen_100.to_array` to convert
the `polars.DataFrame` to a dictionary of numpy arrays.

```pycon

>>> from arctix.dataset.epic_kitchen_100 import to_array
>>> arrays = to_array(data)  # doctest: +SKIP

```

The dictionary contains some regular arrays and masked arrays because sequences have variable
lengths:

```textmate
{'narration': masked_array(
  data=[['open door', 'turn on light', 'close door', ..., --, --, --],
        ['take plate', 'open bin', 'throw leftovers into bin', ..., --,
         --, --],
        ['open door', 'close door', 'switch on lights', ..., --, --, --],
        ...,
        ['open fridge', 'take plastic box', 'close fridge', ..., --, --,
         --],
        ['shake pot', 'take spatula', 'stir chicken thighs in pot', ...,
         --, --, --],
        ['take chicken thighs', 'debone chicken thighs',
         'debone chicken thighs', ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value='N/A',
  dtype='<U77'), 'narration_id': masked_array(
  data=[['P01_01_0', 'P01_01_1', 'P01_01_2', ..., --, --, --],
        ['P01_02_0', 'P01_02_1', 'P01_02_2', ..., --, --, --],
        ['P01_03_0', 'P01_03_1', 'P01_03_2', ..., --, --, --],
        ...,
        ['P37_101_0', 'P37_101_1', 'P37_101_2', ..., --, --, --],
        ['P37_102_0', 'P37_102_1', 'P37_102_2', ..., --, --, --],
        ['P37_103_0', 'P37_103_1', 'P37_103_2', ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value='N/A',
  dtype='<U11'), 'noun': masked_array(
  data=[['door', 'light', 'door', ..., --, --, --],
        ['plate', 'bin', 'leftover', ..., --, --, --],
        ['door', 'door', 'light', ..., --, --, --],
        ...,
        ['fridge', 'box:plastic', 'fridge', ..., --, --, --],
        ['pot', 'spatula', 'thigh:chicken', ..., --, --, --],
        ['thigh:chicken', 'thigh:chicken', 'thigh:chicken', ..., --, --,
         --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value='N/A',
  dtype='<U29'), 'noun_class': masked_array(
  data=[[3, 114, 3, ..., --, --, --],
        [2, 36, 34, ..., --, --, --],
        [3, 3, 114, ..., --, --, --],
        ...,
        [12, 23, 12, ..., --, --, --],
        [29, 20, 57, ..., --, --, --],
        [57, 57, 57, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=999999), 'participant_id': array(['P01', 'P01', 'P01', ..., 'P37', 'P37', 'P37'],
      dtype='<U3'), 'sequence_length': array([329, 145,  42,  ..., 283, 378, 150,
        74]), 'start_frame': masked_array(
  data=[[8, 262, 418, ..., --, --, --],
        [304, 516, 607, ..., --, --, --],
        [16, 195, 292, ..., --, --, --],
        ...,
        [122, 282, 501, ..., --, --, --],
        [215, 388, 472, ..., --, --, --],
        [41, 85, 220, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=999999), 'start_time_second': masked_array(
  data=[[0.13999999999999999, 4.37, 6.9799999999999995, ..., --, --, --],
        [5.069999999999999, 8.61, 10.129999999999999, ..., --, --, --],
        [0.26999999999999996, 3.25, 4.88, ..., --, --, --],
        ...,
        [2.4499999999999997, 5.64, 10.02, ..., --, --, --],
        [4.3, 7.76, 9.45, ..., --, --, --],
        [0.82, 1.7, 4.41, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=1e+20), 'stop_frame': masked_array(
  data=[[202, 370, 569, ..., --, --, --],
        [410, 556, 1087, ..., --, --, --],
        [126, 352, 362, ..., --, --, --],
        ...,
        [301, 500, 621, ..., --, --, --],
        [312, 463, 1154, ..., --, --, --],
        [79, 182, 483, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=999999), 'stop_time_second': masked_array(
  data=[[3.3699999999999997, 6.17, 9.49, ..., --, --, --],
        [6.84, 9.28, 18.13, ..., --, --, --],
        [2.11, 5.88, 6.04, ..., --, --, --],
        ...,
        [6.02, 10.01, 12.43, ..., --, --, --],
        [6.25, 9.26, 23.09, ..., --, --, --],
        [1.5899999999999999, 3.6399999999999997, 9.66, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=1e+20), 'verb': masked_array(
  data=[['open', 'turn-on', 'close', ..., --, --, --],
        ['take', 'open', 'throw-into', ..., --, --, --],
        ['open', 'close', 'switch-on', ..., --, --, --],
        ...,
        ['open', 'take', 'close', ..., --, --, --],
        ['shake', 'take', 'stir-in', ..., --, --, --],
        ['take', 'debone', 'debone', ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value='N/A',
  dtype='<U15'), 'verb_class': masked_array(
  data=[[3, 6, 4, ..., --, --, --],
        [0, 3, 13, ..., --, --, --],
        [3, 4, 6, ..., --, --, --],
        ...,
        [3, 0, 4, ..., --, --, --],
        [15, 0, 10, ..., --, --, --],
        [0, 30, 30, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=999999), 'video_id': array(['P01_01', 'P01_02', 'P01_03', ..., 'P37_101', 'P37_102', 'P37_103'], dtype='<U7')}
```
