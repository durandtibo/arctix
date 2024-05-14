# Ego4D

This page contains information about the Ego4D dataset, which was published with the
following paper:

```textmate
Ego4D: Around the World in 3,000 Hours of Egocentric Video.
K Grauman et al.
arXiv:2110.07058 2021 (https://arxiv.org/abs/2110.07058)
```

The data can be downloaded from
the [official project page](https://ego4d-data.org/docs/).

## Get the raw data

`arctix` provides the function `arctix.dataset.ego4d.fetch_data` to easily load the raw
data in a `polars.DataFrame` format.
Note that you need to download the data by following the instructions on
the [Ego4d website](https://ego4d-data.org/).

```pycon

>>> from pathlib import Path
>>> from arctix.dataset.ego4d import fetch_data
>>> dataset_path = Path("/path/to/dataset/ego4d")
>>> data_raw, metadata_raw = fetch_data(dataset_path, split="train")  # doctest: +SKIP
shape: (63_956, 12)
┌────────────┬────────────┬────────────┬───────────┬───┬───────┬───────────┬───────────┬───────────┐
│ action_cli ┆ action_cli ┆ action_cli ┆ action_cl ┆ … ┆ split ┆ verb      ┆ verb_labe ┆ video_uid │
│ p_end_fram ┆ p_end_sec  ┆ p_start_fr ┆ ip_start_ ┆   ┆ ---   ┆ ---       ┆ l         ┆ ---       │
│ e          ┆ ---        ┆ ame        ┆ sec       ┆   ┆ str   ┆ str       ┆ ---       ┆ str       │
│ ---        ┆ f64        ┆ ---        ┆ ---       ┆   ┆       ┆           ┆ i64       ┆           │
│ i64        ┆            ┆ i64        ┆ f64       ┆   ┆       ┆           ┆           ┆           │
╞════════════╪════════════╪════════════╪═══════════╪═══╪═══════╪═══════════╪═══════════╪═══════════╡
│ 373        ┆ 12.421029  ┆ 133        ┆ 4.4210286 ┆ … ┆ train ┆ put_(plac ┆ 65        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆ e,_leave, ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆ _drop)    ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ 5b3…      │
│ 426        ┆ 14.187695  ┆ 186        ┆ 6.187695  ┆ … ┆ train ┆ take_(pic ┆ 92        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆ k,_grab,_ ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆ get)      ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ 5b3…      │
│ 610        ┆ 20.321029  ┆ 370        ┆ 12.321029 ┆ … ┆ train ┆ move_(tra ┆ 49        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆ nsfer,_pa ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆ ss,_excha ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆ nge…      ┆           ┆ 5b3…      │
│ 537        ┆ 17.887695  ┆ 297        ┆ 9.887695  ┆ … ┆ train ┆ put_(plac ┆ 65        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆ e,_leave, ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆ _drop)    ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ 5b3…      │
│ 550        ┆ 18.321029  ┆ 310        ┆ 10.321029 ┆ … ┆ train ┆ remove    ┆ 67        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ 5b3…      │
│ …          ┆ …          ┆ …          ┆ …         ┆ … ┆ …     ┆ …         ┆ …         ┆ …         │
│ 9142       ┆ 304.721029 ┆ 8902       ┆ 296.72102 ┆ … ┆ train ┆ move_(tra ┆ 49        ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 9         ┆   ┆       ┆ nsfer,_pa ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆ ss,_excha ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆ nge…      ┆           ┆ eab…      │
│ 9199       ┆ 306.621029 ┆ 8959       ┆ 298.62102 ┆ … ┆ train ┆ turn_(spi ┆ 99        ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 9         ┆   ┆       ┆ n,_rotate ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆ ,_flip,_t ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆ urn…      ┆           ┆ eab…      │
│ 9269       ┆ 308.954362 ┆ 9029       ┆ 300.95436 ┆ … ┆ train ┆ carry     ┆ 6         ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 2         ┆   ┆       ┆           ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ eab…      │
│ 9328       ┆ 310.921029 ┆ 9088       ┆ 302.92102 ┆ … ┆ train ┆ turn_(spi ┆ 99        ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 9         ┆   ┆       ┆ n,_rotate ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆ ,_flip,_t ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆ urn…      ┆           ┆ eab…      │
│ 9358       ┆ 311.921029 ┆ 9118       ┆ 303.92102 ┆ … ┆ train ┆ put_(plac ┆ 65        ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 9         ┆   ┆       ┆ e,_leave, ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆ _drop)    ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ eab…      │
└────────────┴────────────┴────────────┴───────────┴───┴───────┴───────────┴───────────┴───────────┘

(vocab_noun): Vocabulary(vocab_size=521)
(vocab_verb): Vocabulary(vocab_size=117)

```

## Prepare the data

`arctix` provides the function `arctix.dataset.ego4d.prepare_data` to preprocess the raw
data.
It returns two outputs: the prepared data and the generate metadata.

```pycon

>>> from arctix.dataset.ego4d import prepare_data
>>> data, metadata = prepare_data(data_raw, metadata_raw)  # doctest: +SKIP

```

`data` is a `polars.DataFrame` which contains the prepared data:

```textmate
shape: (63_956, 12)
┌────────────┬────────────┬────────────┬───────────┬───┬───────┬───────────┬───────────┬───────────┐
│ action_cli ┆ action_cli ┆ action_cli ┆ action_cl ┆ … ┆ split ┆ verb      ┆ verb_labe ┆ video_uid │
│ p_end_fram ┆ p_end_sec  ┆ p_start_fr ┆ ip_start_ ┆   ┆ ---   ┆ ---       ┆ l         ┆ ---       │
│ e          ┆ ---        ┆ ame        ┆ sec       ┆   ┆ str   ┆ str       ┆ ---       ┆ str       │
│ ---        ┆ f64        ┆ ---        ┆ ---       ┆   ┆       ┆           ┆ i64       ┆           │
│ i64        ┆            ┆ i64        ┆ f64       ┆   ┆       ┆           ┆           ┆           │
╞════════════╪════════════╪════════════╪═══════════╪═══╪═══════╪═══════════╪═══════════╪═══════════╡
│ 373        ┆ 12.421029  ┆ 133        ┆ 4.4210286 ┆ … ┆ train ┆ put_(plac ┆ 65        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆ e,_leave, ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆ _drop)    ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ 5b3…      │
│ 426        ┆ 14.187695  ┆ 186        ┆ 6.187695  ┆ … ┆ train ┆ take_(pic ┆ 92        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆ k,_grab,_ ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆ get)      ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ 5b3…      │
│ 610        ┆ 20.321029  ┆ 370        ┆ 12.321029 ┆ … ┆ train ┆ move_(tra ┆ 49        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆ nsfer,_pa ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆ ss,_excha ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆ nge…      ┆           ┆ 5b3…      │
│ 537        ┆ 17.887695  ┆ 297        ┆ 9.887695  ┆ … ┆ train ┆ put_(plac ┆ 65        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆ e,_leave, ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆ _drop)    ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ 5b3…      │
│ 550        ┆ 18.321029  ┆ 310        ┆ 10.321029 ┆ … ┆ train ┆ remove    ┆ 67        ┆ 002d2729- │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ df71-438d │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ -8396-589 │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ 5b3…      │
│ …          ┆ …          ┆ …          ┆ …         ┆ … ┆ …     ┆ …         ┆ …         ┆ …         │
│ 9142       ┆ 304.721029 ┆ 8902       ┆ 296.72102 ┆ … ┆ train ┆ move_(tra ┆ 49        ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 9         ┆   ┆       ┆ nsfer,_pa ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆ ss,_excha ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆ nge…      ┆           ┆ eab…      │
│ 9199       ┆ 306.621029 ┆ 8959       ┆ 298.62102 ┆ … ┆ train ┆ turn_(spi ┆ 99        ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 9         ┆   ┆       ┆ n,_rotate ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆ ,_flip,_t ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆ urn…      ┆           ┆ eab…      │
│ 9269       ┆ 308.954362 ┆ 9029       ┆ 300.95436 ┆ … ┆ train ┆ carry     ┆ 6         ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 2         ┆   ┆       ┆           ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ eab…      │
│ 9328       ┆ 310.921029 ┆ 9088       ┆ 302.92102 ┆ … ┆ train ┆ turn_(spi ┆ 99        ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 9         ┆   ┆       ┆ n,_rotate ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆ ,_flip,_t ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆ urn…      ┆           ┆ eab…      │
│ 9358       ┆ 311.921029 ┆ 9118       ┆ 303.92102 ┆ … ┆ train ┆ put_(plac ┆ 65        ┆ ffb7ecf6- │
│            ┆            ┆            ┆ 9         ┆   ┆       ┆ e,_leave, ┆           ┆ f44e-499b │
│            ┆            ┆            ┆           ┆   ┆       ┆ _drop)    ┆           ┆ -b315-a4a │
│            ┆            ┆            ┆           ┆   ┆       ┆           ┆           ┆ eab…      │
└────────────┴────────────┴────────────┴───────────┴───┴───────┴───────────┴───────────┴───────────┘
```

You can note that new columns are added and the data types of some columns have changed.
`metadata` contains a vocabulary that is used to generate the column `action_id`:

```textmate
{'vocab_noun': Vocabulary(
  counter=Counter({'apple': 1, 'apron': 1, 'arm': 1, 'artwork_(art,_draw,_drawing,_painting,_sketch)': 1, 'asparagus': 1, 'avocado': 1, 'awl': 1, 'axe': 1, 'baby': 1, 'bacon': 1, 'bag_(bag,_grocery,_nylon,_polythene,_pouch,_sachet,_sack,_suitcase)': 1, 'baking_soda': 1, 'ball_(ball,_baseball,_basketball)': 1, 'ball_bearing': 1, 'balloon': 1, 'banana_(banana,_plantain)': 1, 'bar': 1, 'baseboard': 1, 'basket': 1, 'bat_(sports)': 1, 'bat_(tool)': 1, 'bathtub': 1, 'batter_(batter,_mixture)': 1, 'battery': 1, 'bead': 1, 'beaker': 1, 'bean': 1, 'bed': 1, 'belt': 1, 'bench': 1, 'berry': 1, 'beverage_(drink,_juice,_beer,_beverage,_champagne)': 1, 'bicycle_(bicycle,_bike)': 1, 'blanket_(bedsheet,_blanket,_duvet)': 1, 'blender': 1, 'block_(material)': 1, 'blower': 1, 'bolt_extractor': 1, 'book_(book,_booklet,_magazine,_manual,_notebook,_novel,_page,_textbook)': 1, 'bookcase': 1, 'bottle_(bottle,_flask)': 1, 'bowl': 1, 'bracelet_(bangle,_bracelet)': 1, 'brake_(brake,_break)': 1, 'brake_pad': 1, 'branch': 1, 'bread_(bread,_bun,_chapati,_flatbread,_loaf,_roti,_tortilla)': 1, 'brick': 1, 'broccoli': 1, 'broom_(broom,_broomstick)': 1, 'brush': 1, 'bubble_gum': 1, 'bucket': 1, 'buckle': 1, 'burger': 1, 'butter': 1, 'butterfly': 1, 'button': 1, 'cabbage': 1, 'cabinet_(cabinet,_compartment,_cupboard)': 1, 'calculator': 1, 'caliper': 1, 'camera': 1, 'can_opener': 1, 'candle': 1, 'canvas': 1, 'car_(car,_vehicle)': 1, 'card': 1, 'cardboard_(cardboard,_paperboard)': 1, 'carpet': 1, 'carrot': 1, 'cart_(cart,_trolley)': 1, 'cat': 1, 'ceiling': 1, 'celery': 1, 'cello': 1, 'cement_(cement,_concrete,_mortar)': 1, 'cereal': 1, 'chaff': 1, 'chain': 1, 'chair': 1, 'chalk': 1, 'cheese': 1, 'chicken': 1, 'chip_(food)': 1, "chip_(wood'_metal),": 1, 'chip_(wood,_metal)': 1, 'chisel': 1, 'chocolate': 1, 'chopping_board': 1, 'chopstick': 1, 'cigarette_(cigarette,_vape)': 1, 'circuit': 1, 'clamp': 1, 'clay': 1, 'clip': 1, 'clock': 1, 'cloth_(cloth,_fabric,_garment,_kanga,_rag)': 1, 'coaster': 1, 'coconut': 1, 'coffee': 1, 'coffee_machine': 1, 'colander': 1, 'comb': 1, 'computer_(computer,_ipad,_laptop,_motherboard,_screen)': 1, 'container_(box,_can,_carton,_case,_casing,_container,_crate,_holder,_jar,_jerrycan,_keg,_pack,_package,_packaging,_packet,_storage,_tank,_tin)': 1, 'cooker': 1, 'cookie': 1, 'cork': 1, 'corn': 1, 'corner': 1, 'countertop_(counter,_countertop)': 1, 'crab': 1, 'cracker_(biscuit,_cracker)': 1, 'crayon': 1, 'cream': 1, 'crochet': 1, 'crowbar': 1, 'cucumber': 1, 'cup_(cup,_mug,_tumbler)': 1, 'curtain': 1, 'cushion': 1, 'cutter_(tool)': 1, 'decoration_(decoration,_ornament)': 1, 'derailleur': 1, 'detergent': 1, 'dice_(dice,_die)': 1, 'dishwasher': 1, 'dog': 1, 'door': 1, 'doorbell': 1, 'dough': 1, 'dough_mixer': 1, 'doughnut': 1, 'drawer': 1, 'dress': 1, 'drill_(drill,_driller)': 1, 'drill_bit': 1, 'drum': 1, 'dumbbell': 1, 'dust_(dust,_sawdust)': 1, 'duster': 1, 'dustpan': 1, 'egg': 1, 'eggplant': 1, 'engine_(assembly,_carburetor,_engine,_motor)': 1, 'envelope_(envelop,_envelope)': 1, 'eraser_(eraser,_rubber)': 1, 'facemask': 1, 'fan': 1, 'faucet_(faucet,_tap)': 1, 'fence': 1, 'file_(tool)': 1, 'filler': 1, 'filter': 1, 'fish': 1, 'fishing_rod': 1, 'flash_drive': 1, 'floor_(floor,_ground)': 1, 'flour': 1, 'flower': 1, 'foam': 1, 'foil': 1, 'food': 1, 'foot_(foot,_toe)': 1, 'fork': 1, 'fridge_(fridge,_refrigerator)': 1, 'fries': 1, 'fuel': 1, 'funnel': 1, 'game_controller': 1, 'garbage_can_(bin,_dustbin)': 1, 'garlic': 1, 'gasket': 1, 'gate': 1, 'gauge': 1, 'gauze': 1, 'gear': 1, 'generator': 1, 'ginger': 1, 'glass': 1, 'glasses_(goggle,_shade,_spectacle,_sunglass)': 1, 'glove': 1, 'glue_(adhesive,_glue,_gum,_sealant)': 1, 'glue_gun': 1, 'golf_club': 1, 'gourd': 1, 'grain': 1, 'grape': 1, 'grapefruit': 1, 'grass': 1, 'grater': 1, 'grill': 1, 'grinder': 1, 'guava': 1, 'guitar': 1, 'hair': 1, 'hammer_(hammer,_mallet)': 1, 'hand_(finger,_hand,_palm,_thumb)': 1, 'handle': 1, 'hanger': 1, 'hat': 1, 'hay': 1, 'haystack': 1, 'head': 1, 'headphones_(earphone,_headphone)': 1, 'heater': 1, 'helmet': 1, 'hinge': 1, 'hole': 1, 'horse': 1, 'hose': 1, 'house': 1, 'ice': 1, 'ice_cream': 1, 'ink': 1, 'iron': 1, 'jack_(tool)_(jack,_lift)': 1, 'jacket_(coat,_jacket)': 1, 'jug': 1, 'kale': 1, 'ketchup': 1, 'kettle': 1, 'key': 1, 'keyboard': 1, 'knife_(knife,_machete)': 1, 'label_(label,_tag)': 1, 'ladder': 1, 'leaf_(leaf,_leave)': 1, 'leash': 1, 'leg_(knee,_leg,_thigh)': 1, 'lemon': 1, 'lever': 1, 'lid_(cap,_cover,_lid)': 1, 'light_(bulb,_flashlight,_lamp,_light)': 1, 'lighter': 1, 'lime': 1, 'lock': 1, 'lubricant_(grease,_lubricant)': 1, 'magnet_(magnet,_sphere)': 1, 'mango': 1, 'manure_(dung,_manure)': 1, 'mask': 1, 'mat_(mat,_rug)': 1, 'matchstick': 1, 'meat_(beef,_ham,_meat)': 1, 'medicine': 1, 'metal_(lead,_metal,_steel)': 1, 'microscope': 1, 'microwave': 1, 'milk': 1, 'mirror': 1, 'mixer': 1, 'mold_(mold,_molder,_mould)': 1, 'money_(cash,_coin,_money)': 1, 'mop': 1, 'motorcycle_(motorbike,_motorcycle)': 1, 'mouse_(computer)': 1, 'mouth': 1, 'mower_(lawnmower,_mower)': 1, 'multimeter': 1, 'mushroom': 1, 'nail_cutter': 1, 'nail_gun': 1, 'nail_polish': 1, 'napkin_(handkerchief,_napkin,_serviette,_tissue,_wipe)': 1, 'necklace': 1, 'needle_(hook,_needle)': 1, 'net': 1, 'nozzle': 1, 'nut_(food)': 1, 'nut_(tool)': 1, 'oil_(fat,_oil)': 1, 'okra': 1, 'onion': 1, 'oven': 1, 'paddle': 1, 'paint': 1, 'paint_roller': 1, 'paintbrush': 1, 'palette': 1, 'pan_(frypan,_pan,_saucepan)': 1, 'pancake': 1, 'panel': 1, 'pants_(jean,_pant,_short,_trouser)': 1, 'papaya': 1, 'paper_(chart,_craft,_newspaper,_note,_paper,_papercraft,_poster,_receipt)': 1, 'pasta_(noodle,_pasta,_spaghetti)': 1, 'paste': 1, 'pastry': 1, 'pea': 1, 'peanut': 1, 'pear': 1, 'pedal': 1, 'peel': 1, 'peeler': 1, 'peg': 1, 'pen_(marker,_pen)': 1, 'pencil': 1, 'pepper_(vegetable)_(capsicum,_pepper)': 1, 'phone_(cellphone,_phone,_smartphone)': 1, 'photo': 1, 'piano': 1, 'pickle': 1, 'picture_(picture,_portrait)': 1, 'pie': 1, 'pillow': 1, 'pilot_jet': 1, 'pin': 1, 'pipe': 1, 'pizza': 1, 'planer_(plane,_planer)': 1, 'plant_(bud,_frond,_plant,_reed,_seedling,_shrub,_stem,_vine,_weed)': 1, 'plate_(dish,_plate,_platter,_saucer)': 1, 'playing_cards': 1, 'plier': 1, 'plug': 1, 'pole': 1, 'popcorn': 1, 'pot': 1, 'pot_(planter)': 1, 'potato': 1, 'pump': 1, 'pumpkin': 1, 'purse': 1, 'puzzle_or_game_piece_(chess,_domino,_jenga,_jigsaw,_pawn,_puzzle)': 1, 'rack': 1, 'radio': 1, 'rail_(rail,_railing)': 1, 'rake': 1, 'razor_blade': 1, 'remote_control_(control,_remote)': 1, 'rice': 1, 'ring': 1, 'rod_(dipstick,_rod,_rod_metal,_shaft)': 1, 'rolling_pin': 1, 'root': 1, 'rope': 1, 'router': 1, 'rubber_band': 1, 'ruler_(rule,_ruler)': 1, 'sand': 1, 'sander': 1, 'sandpaper': 1, 'sandwich': 1, 'sauce': 1, 'sausage': 1, 'saw_(chainsaw,_saw,_hacksaw)': 1, 'scarf_(scarf,_shawl)': 1, 'scissors': 1, 'scoop_(scoop,_scooper)': 1, 'scraper_(scraper,_scrapper)': 1, 'screw_(bolt,_nail,_screw)': 1, 'screwdriver': 1, 'sculpture': 1, 'seasoning_(salt,_seasoning,_shaker,_spice,_sugar)': 1, 'seed': 1, 'set_square_(tool)': 1, 'sewing_machine': 1, 'sharpener': 1, 'shears': 1, 'sheet': 1, 'shelf': 1, 'shell_(egg_shell,_shell_egg)': 1, 'shirt_(cardigan,_shirt,_sweater,_sweatshirt,_top)': 1, 'shoe_(boot,_sandal,_shoe,_slipper)': 1, 'shovel_(hoe,_shovel,_spade)': 1, 'shower_head': 1, 'sickle': 1, 'sieve_(sieve,_strainer)': 1, 'sink_(basin,_sink)': 1, 'sketch_pad': 1, 'skirt': 1, 'slab': 1, 'snorkel': 1, 'soap': 1, 'sock': 1, 'socket': 1, 'sofa': 1, 'soil_(dirt,_mud,_soil)': 1, 'solder_iron': 1, 'soup': 1, 'spacer': 1, 'spatula': 1, 'speaker': 1, 'sphygmomanometer': 1, 'spice': 1, 'spinach': 1, 'spirit_level': 1, 'sponge_(scrubber,_sponge)': 1, 'spoon_(spoon,_spoonful)': 1, 'spray_(spray,_sprayer)': 1, 'spring': 1, 'squeezer': 1, 'stairs_(stair,_staircase)': 1, 'stamp': 1, 'stapler': 1, 'steamer': 1, 'steering_wheel': 1, 'stick_(stick,_twig)': 1, 'sticker': 1, 'stock_(food)': 1, 'stone_(rock,_stone)': 1, 'stool': 1, 'stove_(burner,_gas,_stove)': 1, 'strap': 1, 'straw': 1, 'string_(bobbin,_knot,_lace,_ribbon,_spool,_strand,_string,_thread,_twine,_wool,_yarn)': 1, 'stroller': 1, 'switch_(knob,_switch)': 1, 'syringe': 1, 'table_(stand,_table)': 1, 'tablet': 1, 'taco': 1, 'tape_(cellotape,_sellotape,_tape)': 1, 'tape_measure_(measure,_measurement)': 1, 'tea': 1, 'teapot': 1, 'television_(television,_tv)': 1, 'tent': 1, 'test_tube': 1, 'tie': 1, 'tile': 1, 'timer': 1, 'toaster': 1, 'toilet': 1, 'toilet_paper': 1, 'tomato': 1, 'tongs': 1, 'toolbox': 1, 'toothbrush': 1, 'toothpick': 1, 'torch_(torch,_torchlight)': 1, 'towel': 1, 'toy_(doll,_toy)': 1, 'tractor': 1, 'trash_(debris,_garbage,_litter,_trash,_waste)': 1, 'tray': 1, 'treadmill': 1, 'tree': 1, 'trimmer_(pruner,_trimmer)': 1, 'trowel': 1, 'truck': 1, 'tweezer': 1, 'umbrella': 1, 'undergarment_(boxer,_bra)': 1, 'vacuum': 1, 'vacuum_cleaner': 1, 'valve': 1, 'vase': 1, 'video_game': 1, 'violin': 1, 'wall': 1, 'wallet': 1, 'wallpaper': 1, 'washing_machine': 1, 'watch': 1, 'water': 1, 'watermelon': 1, 'weighing_scale': 1, 'welding_torch': 1, 'wheat_(maize,_wheat)': 1, 'wheel_(tyre,_wheel)': 1, 'wheelbarrow': 1, 'whisk': 1, 'window': 1, 'windshield': 1, 'wiper_(car)': 1, 'wire_(adapter,_cable,_charger,_connector,_cord,_wire)': 1, 'wood_(fiber,_firewood,_floorboard,_log,_lumber,_plank,_plywood,_timber,_wood,_woodcraft,_woodwork)': 1, 'worm': 1, 'wrapper_(covering,_film,_seal,_wrap,_wrapper,_wrapping)': 1, 'wrench_(spanner,_wrench)': 1, 'yam': 1, 'yeast': 1, 'yoghurt': 1, 'zipper_(zip,_zipper)': 1, 'zucchini': 1, 'ambulance': 1, 'back': 1, 'bamboo': 1, 'bandage': 1, 'baton': 1, 'bird': 1, 'brownie': 1, 'cake': 1, 'cash_register': 1, 'cassava': 1, 'cocoa': 1, 'courgette': 1, 'cow': 1, 'cupcake': 1, 'drone': 1, 'earplug': 1, 'hotdog': 1, 'juicer': 1, 'kiwi': 1, 'ladle': 1, 'leek': 1, 'lettuce': 1, 'marble': 1, 'melon': 1, 'orange': 1, 'peach': 1, 'person_(herself,_himself,_lady,_man,_person,_shoulder,_they,_woman)': 1, 'pipette': 1, 'plum': 1, 'plunger': 1, 'printer': 1, 'putty': 1, 'racket': 1, 'ratchet': 1, 'road': 1, 'salad': 1, 'scaffold': 1, 'squash': 1, 'stereo': 1, 'strawberry': 1, 'thermometer': 1, 'transistor': 1, 'vinegar': 1}),
  index_to_token=('apple', 'apron', 'arm', 'artwork_(art,_draw,_drawing,_painting,_sketch)', 'asparagus', 'avocado', 'awl', 'axe', 'baby', 'bacon', 'bag_(bag,_grocery,_nylon,_polythene,_pouch,_sachet,_sack,_suitcase)', 'baking_soda', 'ball_(ball,_baseball,_basketball)', 'ball_bearing', 'balloon', 'banana_(banana,_plantain)', 'bar', 'baseboard', 'basket', 'bat_(sports)', 'bat_(tool)', 'bathtub', 'batter_(batter,_mixture)', 'battery', 'bead', 'beaker', 'bean', 'bed', 'belt', 'bench', 'berry', 'beverage_(drink,_juice,_beer,_beverage,_champagne)', 'bicycle_(bicycle,_bike)', 'blanket_(bedsheet,_blanket,_duvet)', 'blender', 'block_(material)', 'blower', 'bolt_extractor', 'book_(book,_booklet,_magazine,_manual,_notebook,_novel,_page,_textbook)', 'bookcase', 'bottle_(bottle,_flask)', 'bowl', 'bracelet_(bangle,_bracelet)', 'brake_(brake,_break)', 'brake_pad', 'branch', 'bread_(bread,_bun,_chapati,_flatbread,_loaf,_roti,_tortilla)', 'brick', 'broccoli', 'broom_(broom,_broomstick)', 'brush', 'bubble_gum', 'bucket', 'buckle', 'burger', 'butter', 'butterfly', 'button', 'cabbage', 'cabinet_(cabinet,_compartment,_cupboard)', 'calculator', 'caliper', 'camera', 'can_opener', 'candle', 'canvas', 'car_(car,_vehicle)', 'card', 'cardboard_(cardboard,_paperboard)', 'carpet', 'carrot', 'cart_(cart,_trolley)', 'cat', 'ceiling', 'celery', 'cello', 'cement_(cement,_concrete,_mortar)', 'cereal', 'chaff', 'chain', 'chair', 'chalk', 'cheese', 'chicken', 'chip_(food)', "chip_(wood'_metal),", 'chip_(wood,_metal)', 'chisel', 'chocolate', 'chopping_board', 'chopstick', 'cigarette_(cigarette,_vape)', 'circuit', 'clamp', 'clay', 'clip', 'clock', 'cloth_(cloth,_fabric,_garment,_kanga,_rag)', 'coaster', 'coconut', 'coffee', 'coffee_machine', 'colander', 'comb', 'computer_(computer,_ipad,_laptop,_motherboard,_screen)', 'container_(box,_can,_carton,_case,_casing,_container,_crate,_holder,_jar,_jerrycan,_keg,_pack,_package,_packaging,_packet,_storage,_tank,_tin)', 'cooker', 'cookie', 'cork', 'corn', 'corner', 'countertop_(counter,_countertop)', 'crab', 'cracker_(biscuit,_cracker)', 'crayon', 'cream', 'crochet', 'crowbar', 'cucumber', 'cup_(cup,_mug,_tumbler)', 'curtain', 'cushion', 'cutter_(tool)', 'decoration_(decoration,_ornament)', 'derailleur', 'detergent', 'dice_(dice,_die)', 'dishwasher', 'dog', 'door', 'doorbell', 'dough', 'dough_mixer', 'doughnut', 'drawer', 'dress', 'drill_(drill,_driller)', 'drill_bit', 'drum', 'dumbbell', 'dust_(dust,_sawdust)', 'duster', 'dustpan', 'egg', 'eggplant', 'engine_(assembly,_carburetor,_engine,_motor)', 'envelope_(envelop,_envelope)', 'eraser_(eraser,_rubber)', 'facemask', 'fan', 'faucet_(faucet,_tap)', 'fence', 'file_(tool)', 'filler', 'filter', 'fish', 'fishing_rod', 'flash_drive', 'floor_(floor,_ground)', 'flour', 'flower', 'foam', 'foil', 'food', 'foot_(foot,_toe)', 'fork', 'fridge_(fridge,_refrigerator)', 'fries', 'fuel', 'funnel', 'game_controller', 'garbage_can_(bin,_dustbin)', 'garlic', 'gasket', 'gate', 'gauge', 'gauze', 'gear', 'generator', 'ginger', 'glass', 'glasses_(goggle,_shade,_spectacle,_sunglass)', 'glove', 'glue_(adhesive,_glue,_gum,_sealant)', 'glue_gun', 'golf_club', 'gourd', 'grain', 'grape', 'grapefruit', 'grass', 'grater', 'grill', 'grinder', 'guava', 'guitar', 'hair', 'hammer_(hammer,_mallet)', 'hand_(finger,_hand,_palm,_thumb)', 'handle', 'hanger', 'hat', 'hay', 'haystack', 'head', 'headphones_(earphone,_headphone)', 'heater', 'helmet', 'hinge', 'hole', 'horse', 'hose', 'house', 'ice', 'ice_cream', 'ink', 'iron', 'jack_(tool)_(jack,_lift)', 'jacket_(coat,_jacket)', 'jug', 'kale', 'ketchup', 'kettle', 'key', 'keyboard', 'knife_(knife,_machete)', 'label_(label,_tag)', 'ladder', 'leaf_(leaf,_leave)', 'leash', 'leg_(knee,_leg,_thigh)', 'lemon', 'lever', 'lid_(cap,_cover,_lid)', 'light_(bulb,_flashlight,_lamp,_light)', 'lighter', 'lime', 'lock', 'lubricant_(grease,_lubricant)', 'magnet_(magnet,_sphere)', 'mango', 'manure_(dung,_manure)', 'mask', 'mat_(mat,_rug)', 'matchstick', 'meat_(beef,_ham,_meat)', 'medicine', 'metal_(lead,_metal,_steel)', 'microscope', 'microwave', 'milk', 'mirror', 'mixer', 'mold_(mold,_molder,_mould)', 'money_(cash,_coin,_money)', 'mop', 'motorcycle_(motorbike,_motorcycle)', 'mouse_(computer)', 'mouth', 'mower_(lawnmower,_mower)', 'multimeter', 'mushroom', 'nail_cutter', 'nail_gun', 'nail_polish', 'napkin_(handkerchief,_napkin,_serviette,_tissue,_wipe)', 'necklace', 'needle_(hook,_needle)', 'net', 'nozzle', 'nut_(food)', 'nut_(tool)', 'oil_(fat,_oil)', 'okra', 'onion', 'oven', 'paddle', 'paint', 'paint_roller', 'paintbrush', 'palette', 'pan_(frypan,_pan,_saucepan)', 'pancake', 'panel', 'pants_(jean,_pant,_short,_trouser)', 'papaya', 'paper_(chart,_craft,_newspaper,_note,_paper,_papercraft,_poster,_receipt)', 'pasta_(noodle,_pasta,_spaghetti)', 'paste', 'pastry', 'pea', 'peanut', 'pear', 'pedal', 'peel', 'peeler', 'peg', 'pen_(marker,_pen)', 'pencil', 'pepper_(vegetable)_(capsicum,_pepper)', 'phone_(cellphone,_phone,_smartphone)', 'photo', 'piano', 'pickle', 'picture_(picture,_portrait)', 'pie', 'pillow', 'pilot_jet', 'pin', 'pipe', 'pizza', 'planer_(plane,_planer)', 'plant_(bud,_frond,_plant,_reed,_seedling,_shrub,_stem,_vine,_weed)', 'plate_(dish,_plate,_platter,_saucer)', 'playing_cards', 'plier', 'plug', 'pole', 'popcorn', 'pot', 'pot_(planter)', 'potato', 'pump', 'pumpkin', 'purse', 'puzzle_or_game_piece_(chess,_domino,_jenga,_jigsaw,_pawn,_puzzle)', 'rack', 'radio', 'rail_(rail,_railing)', 'rake', 'razor_blade', 'remote_control_(control,_remote)', 'rice', 'ring', 'rod_(dipstick,_rod,_rod_metal,_shaft)', 'rolling_pin', 'root', 'rope', 'router', 'rubber_band', 'ruler_(rule,_ruler)', 'sand', 'sander', 'sandpaper', 'sandwich', 'sauce', 'sausage', 'saw_(chainsaw,_saw,_hacksaw)', 'scarf_(scarf,_shawl)', 'scissors', 'scoop_(scoop,_scooper)', 'scraper_(scraper,_scrapper)', 'screw_(bolt,_nail,_screw)', 'screwdriver', 'sculpture', 'seasoning_(salt,_seasoning,_shaker,_spice,_sugar)', 'seed', 'set_square_(tool)', 'sewing_machine', 'sharpener', 'shears', 'sheet', 'shelf', 'shell_(egg_shell,_shell_egg)', 'shirt_(cardigan,_shirt,_sweater,_sweatshirt,_top)', 'shoe_(boot,_sandal,_shoe,_slipper)', 'shovel_(hoe,_shovel,_spade)', 'shower_head', 'sickle', 'sieve_(sieve,_strainer)', 'sink_(basin,_sink)', 'sketch_pad', 'skirt', 'slab', 'snorkel', 'soap', 'sock', 'socket', 'sofa', 'soil_(dirt,_mud,_soil)', 'solder_iron', 'soup', 'spacer', 'spatula', 'speaker', 'sphygmomanometer', 'spice', 'spinach', 'spirit_level', 'sponge_(scrubber,_sponge)', 'spoon_(spoon,_spoonful)', 'spray_(spray,_sprayer)', 'spring', 'squeezer', 'stairs_(stair,_staircase)', 'stamp', 'stapler', 'steamer', 'steering_wheel', 'stick_(stick,_twig)', 'sticker', 'stock_(food)', 'stone_(rock,_stone)', 'stool', 'stove_(burner,_gas,_stove)', 'strap', 'straw', 'string_(bobbin,_knot,_lace,_ribbon,_spool,_strand,_string,_thread,_twine,_wool,_yarn)', 'stroller', 'switch_(knob,_switch)', 'syringe', 'table_(stand,_table)', 'tablet', 'taco', 'tape_(cellotape,_sellotape,_tape)', 'tape_measure_(measure,_measurement)', 'tea', 'teapot', 'television_(television,_tv)', 'tent', 'test_tube', 'tie', 'tile', 'timer', 'toaster', 'toilet', 'toilet_paper', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpick', 'torch_(torch,_torchlight)', 'towel', 'toy_(doll,_toy)', 'tractor', 'trash_(debris,_garbage,_litter,_trash,_waste)', 'tray', 'treadmill', 'tree', 'trimmer_(pruner,_trimmer)', 'trowel', 'truck', 'tweezer', 'umbrella', 'undergarment_(boxer,_bra)', 'vacuum', 'vacuum_cleaner', 'valve', 'vase', 'video_game', 'violin', 'wall', 'wallet', 'wallpaper', 'washing_machine', 'watch', 'water', 'watermelon', 'weighing_scale', 'welding_torch', 'wheat_(maize,_wheat)', 'wheel_(tyre,_wheel)', 'wheelbarrow', 'whisk', 'window', 'windshield', 'wiper_(car)', 'wire_(adapter,_cable,_charger,_connector,_cord,_wire)', 'wood_(fiber,_firewood,_floorboard,_log,_lumber,_plank,_plywood,_timber,_wood,_woodcraft,_woodwork)', 'worm', 'wrapper_(covering,_film,_seal,_wrap,_wrapper,_wrapping)', 'wrench_(spanner,_wrench)', 'yam', 'yeast', 'yoghurt', 'zipper_(zip,_zipper)', 'zucchini', 'ambulance', 'back', 'bamboo', 'bandage', 'baton', 'bird', 'brownie', 'cake', 'cash_register', 'cassava', 'cocoa', 'courgette', 'cow', 'cupcake', 'drone', 'earplug', 'hotdog', 'juicer', 'kiwi', 'ladle', 'leek', 'lettuce', 'marble', 'melon', 'orange', 'peach', 'person_(herself,_himself,_lady,_man,_person,_shoulder,_they,_woman)', 'pipette', 'plum', 'plunger', 'printer', 'putty', 'racket', 'ratchet', 'road', 'salad', 'scaffold', 'squash', 'stereo', 'strawberry', 'thermometer', 'transistor', 'vinegar'),
  token_to_index={'apple': 0, 'apron': 1, 'arm': 2, 'artwork_(art,_draw,_drawing,_painting,_sketch)': 3, 'asparagus': 4, 'avocado': 5, 'awl': 6, 'axe': 7, 'baby': 8, 'bacon': 9, 'bag_(bag,_grocery,_nylon,_polythene,_pouch,_sachet,_sack,_suitcase)': 10, 'baking_soda': 11, 'ball_(ball,_baseball,_basketball)': 12, 'ball_bearing': 13, 'balloon': 14, 'banana_(banana,_plantain)': 15, 'bar': 16, 'baseboard': 17, 'basket': 18, 'bat_(sports)': 19, 'bat_(tool)': 20, 'bathtub': 21, 'batter_(batter,_mixture)': 22, 'battery': 23, 'bead': 24, 'beaker': 25, 'bean': 26, 'bed': 27, 'belt': 28, 'bench': 29, 'berry': 30, 'beverage_(drink,_juice,_beer,_beverage,_champagne)': 31, 'bicycle_(bicycle,_bike)': 32, 'blanket_(bedsheet,_blanket,_duvet)': 33, 'blender': 34, 'block_(material)': 35, 'blower': 36, 'bolt_extractor': 37, 'book_(book,_booklet,_magazine,_manual,_notebook,_novel,_page,_textbook)': 38, 'bookcase': 39, 'bottle_(bottle,_flask)': 40, 'bowl': 41, 'bracelet_(bangle,_bracelet)': 42, 'brake_(brake,_break)': 43, 'brake_pad': 44, 'branch': 45, 'bread_(bread,_bun,_chapati,_flatbread,_loaf,_roti,_tortilla)': 46, 'brick': 47, 'broccoli': 48, 'broom_(broom,_broomstick)': 49, 'brush': 50, 'bubble_gum': 51, 'bucket': 52, 'buckle': 53, 'burger': 54, 'butter': 55, 'butterfly': 56, 'button': 57, 'cabbage': 58, 'cabinet_(cabinet,_compartment,_cupboard)': 59, 'calculator': 60, 'caliper': 61, 'camera': 62, 'can_opener': 63, 'candle': 64, 'canvas': 65, 'car_(car,_vehicle)': 66, 'card': 67, 'cardboard_(cardboard,_paperboard)': 68, 'carpet': 69, 'carrot': 70, 'cart_(cart,_trolley)': 71, 'cat': 72, 'ceiling': 73, 'celery': 74, 'cello': 75, 'cement_(cement,_concrete,_mortar)': 76, 'cereal': 77, 'chaff': 78, 'chain': 79, 'chair': 80, 'chalk': 81, 'cheese': 82, 'chicken': 83, 'chip_(food)': 84, "chip_(wood'_metal),": 85, 'chip_(wood,_metal)': 86, 'chisel': 87, 'chocolate': 88, 'chopping_board': 89, 'chopstick': 90, 'cigarette_(cigarette,_vape)': 91, 'circuit': 92, 'clamp': 93, 'clay': 94, 'clip': 95, 'clock': 96, 'cloth_(cloth,_fabric,_garment,_kanga,_rag)': 97, 'coaster': 98, 'coconut': 99, 'coffee': 100, 'coffee_machine': 101, 'colander': 102, 'comb': 103, 'computer_(computer,_ipad,_laptop,_motherboard,_screen)': 104, 'container_(box,_can,_carton,_case,_casing,_container,_crate,_holder,_jar,_jerrycan,_keg,_pack,_package,_packaging,_packet,_storage,_tank,_tin)': 105, 'cooker': 106, 'cookie': 107, 'cork': 108, 'corn': 109, 'corner': 110, 'countertop_(counter,_countertop)': 111, 'crab': 112, 'cracker_(biscuit,_cracker)': 113, 'crayon': 114, 'cream': 115, 'crochet': 116, 'crowbar': 117, 'cucumber': 118, 'cup_(cup,_mug,_tumbler)': 119, 'curtain': 120, 'cushion': 121, 'cutter_(tool)': 122, 'decoration_(decoration,_ornament)': 123, 'derailleur': 124, 'detergent': 125, 'dice_(dice,_die)': 126, 'dishwasher': 127, 'dog': 128, 'door': 129, 'doorbell': 130, 'dough': 131, 'dough_mixer': 132, 'doughnut': 133, 'drawer': 134, 'dress': 135, 'drill_(drill,_driller)': 136, 'drill_bit': 137, 'drum': 138, 'dumbbell': 139, 'dust_(dust,_sawdust)': 140, 'duster': 141, 'dustpan': 142, 'egg': 143, 'eggplant': 144, 'engine_(assembly,_carburetor,_engine,_motor)': 145, 'envelope_(envelop,_envelope)': 146, 'eraser_(eraser,_rubber)': 147, 'facemask': 148, 'fan': 149, 'faucet_(faucet,_tap)': 150, 'fence': 151, 'file_(tool)': 152, 'filler': 153, 'filter': 154, 'fish': 155, 'fishing_rod': 156, 'flash_drive': 157, 'floor_(floor,_ground)': 158, 'flour': 159, 'flower': 160, 'foam': 161, 'foil': 162, 'food': 163, 'foot_(foot,_toe)': 164, 'fork': 165, 'fridge_(fridge,_refrigerator)': 166, 'fries': 167, 'fuel': 168, 'funnel': 169, 'game_controller': 170, 'garbage_can_(bin,_dustbin)': 171, 'garlic': 172, 'gasket': 173, 'gate': 174, 'gauge': 175, 'gauze': 176, 'gear': 177, 'generator': 178, 'ginger': 179, 'glass': 180, 'glasses_(goggle,_shade,_spectacle,_sunglass)': 181, 'glove': 182, 'glue_(adhesive,_glue,_gum,_sealant)': 183, 'glue_gun': 184, 'golf_club': 185, 'gourd': 186, 'grain': 187, 'grape': 188, 'grapefruit': 189, 'grass': 190, 'grater': 191, 'grill': 192, 'grinder': 193, 'guava': 194, 'guitar': 195, 'hair': 196, 'hammer_(hammer,_mallet)': 197, 'hand_(finger,_hand,_palm,_thumb)': 198, 'handle': 199, 'hanger': 200, 'hat': 201, 'hay': 202, 'haystack': 203, 'head': 204, 'headphones_(earphone,_headphone)': 205, 'heater': 206, 'helmet': 207, 'hinge': 208, 'hole': 209, 'horse': 210, 'hose': 211, 'house': 212, 'ice': 213, 'ice_cream': 214, 'ink': 215, 'iron': 216, 'jack_(tool)_(jack,_lift)': 217, 'jacket_(coat,_jacket)': 218, 'jug': 219, 'kale': 220, 'ketchup': 221, 'kettle': 222, 'key': 223, 'keyboard': 224, 'knife_(knife,_machete)': 225, 'label_(label,_tag)': 226, 'ladder': 227, 'leaf_(leaf,_leave)': 228, 'leash': 229, 'leg_(knee,_leg,_thigh)': 230, 'lemon': 231, 'lever': 232, 'lid_(cap,_cover,_lid)': 233, 'light_(bulb,_flashlight,_lamp,_light)': 234, 'lighter': 235, 'lime': 236, 'lock': 237, 'lubricant_(grease,_lubricant)': 238, 'magnet_(magnet,_sphere)': 239, 'mango': 240, 'manure_(dung,_manure)': 241, 'mask': 242, 'mat_(mat,_rug)': 243, 'matchstick': 244, 'meat_(beef,_ham,_meat)': 245, 'medicine': 246, 'metal_(lead,_metal,_steel)': 247, 'microscope': 248, 'microwave': 249, 'milk': 250, 'mirror': 251, 'mixer': 252, 'mold_(mold,_molder,_mould)': 253, 'money_(cash,_coin,_money)': 254, 'mop': 255, 'motorcycle_(motorbike,_motorcycle)': 256, 'mouse_(computer)': 257, 'mouth': 258, 'mower_(lawnmower,_mower)': 259, 'multimeter': 260, 'mushroom': 261, 'nail_cutter': 262, 'nail_gun': 263, 'nail_polish': 264, 'napkin_(handkerchief,_napkin,_serviette,_tissue,_wipe)': 265, 'necklace': 266, 'needle_(hook,_needle)': 267, 'net': 268, 'nozzle': 269, 'nut_(food)': 270, 'nut_(tool)': 271, 'oil_(fat,_oil)': 272, 'okra': 273, 'onion': 274, 'oven': 275, 'paddle': 276, 'paint': 277, 'paint_roller': 278, 'paintbrush': 279, 'palette': 280, 'pan_(frypan,_pan,_saucepan)': 281, 'pancake': 282, 'panel': 283, 'pants_(jean,_pant,_short,_trouser)': 284, 'papaya': 285, 'paper_(chart,_craft,_newspaper,_note,_paper,_papercraft,_poster,_receipt)': 286, 'pasta_(noodle,_pasta,_spaghetti)': 287, 'paste': 288, 'pastry': 289, 'pea': 290, 'peanut': 291, 'pear': 292, 'pedal': 293, 'peel': 294, 'peeler': 295, 'peg': 296, 'pen_(marker,_pen)': 297, 'pencil': 298, 'pepper_(vegetable)_(capsicum,_pepper)': 299, 'phone_(cellphone,_phone,_smartphone)': 300, 'photo': 301, 'piano': 302, 'pickle': 303, 'picture_(picture,_portrait)': 304, 'pie': 305, 'pillow': 306, 'pilot_jet': 307, 'pin': 308, 'pipe': 309, 'pizza': 310, 'planer_(plane,_planer)': 311, 'plant_(bud,_frond,_plant,_reed,_seedling,_shrub,_stem,_vine,_weed)': 312, 'plate_(dish,_plate,_platter,_saucer)': 313, 'playing_cards': 314, 'plier': 315, 'plug': 316, 'pole': 317, 'popcorn': 318, 'pot': 319, 'pot_(planter)': 320, 'potato': 321, 'pump': 322, 'pumpkin': 323, 'purse': 324, 'puzzle_or_game_piece_(chess,_domino,_jenga,_jigsaw,_pawn,_puzzle)': 325, 'rack': 326, 'radio': 327, 'rail_(rail,_railing)': 328, 'rake': 329, 'razor_blade': 330, 'remote_control_(control,_remote)': 331, 'rice': 332, 'ring': 333, 'rod_(dipstick,_rod,_rod_metal,_shaft)': 334, 'rolling_pin': 335, 'root': 336, 'rope': 337, 'router': 338, 'rubber_band': 339, 'ruler_(rule,_ruler)': 340, 'sand': 341, 'sander': 342, 'sandpaper': 343, 'sandwich': 344, 'sauce': 345, 'sausage': 346, 'saw_(chainsaw,_saw,_hacksaw)': 347, 'scarf_(scarf,_shawl)': 348, 'scissors': 349, 'scoop_(scoop,_scooper)': 350, 'scraper_(scraper,_scrapper)': 351, 'screw_(bolt,_nail,_screw)': 352, 'screwdriver': 353, 'sculpture': 354, 'seasoning_(salt,_seasoning,_shaker,_spice,_sugar)': 355, 'seed': 356, 'set_square_(tool)': 357, 'sewing_machine': 358, 'sharpener': 359, 'shears': 360, 'sheet': 361, 'shelf': 362, 'shell_(egg_shell,_shell_egg)': 363, 'shirt_(cardigan,_shirt,_sweater,_sweatshirt,_top)': 364, 'shoe_(boot,_sandal,_shoe,_slipper)': 365, 'shovel_(hoe,_shovel,_spade)': 366, 'shower_head': 367, 'sickle': 368, 'sieve_(sieve,_strainer)': 369, 'sink_(basin,_sink)': 370, 'sketch_pad': 371, 'skirt': 372, 'slab': 373, 'snorkel': 374, 'soap': 375, 'sock': 376, 'socket': 377, 'sofa': 378, 'soil_(dirt,_mud,_soil)': 379, 'solder_iron': 380, 'soup': 381, 'spacer': 382, 'spatula': 383, 'speaker': 384, 'sphygmomanometer': 385, 'spice': 386, 'spinach': 387, 'spirit_level': 388, 'sponge_(scrubber,_sponge)': 389, 'spoon_(spoon,_spoonful)': 390, 'spray_(spray,_sprayer)': 391, 'spring': 392, 'squeezer': 393, 'stairs_(stair,_staircase)': 394, 'stamp': 395, 'stapler': 396, 'steamer': 397, 'steering_wheel': 398, 'stick_(stick,_twig)': 399, 'sticker': 400, 'stock_(food)': 401, 'stone_(rock,_stone)': 402, 'stool': 403, 'stove_(burner,_gas,_stove)': 404, 'strap': 405, 'straw': 406, 'string_(bobbin,_knot,_lace,_ribbon,_spool,_strand,_string,_thread,_twine,_wool,_yarn)': 407, 'stroller': 408, 'switch_(knob,_switch)': 409, 'syringe': 410, 'table_(stand,_table)': 411, 'tablet': 412, 'taco': 413, 'tape_(cellotape,_sellotape,_tape)': 414, 'tape_measure_(measure,_measurement)': 415, 'tea': 416, 'teapot': 417, 'television_(television,_tv)': 418, 'tent': 419, 'test_tube': 420, 'tie': 421, 'tile': 422, 'timer': 423, 'toaster': 424, 'toilet': 425, 'toilet_paper': 426, 'tomato': 427, 'tongs': 428, 'toolbox': 429, 'toothbrush': 430, 'toothpick': 431, 'torch_(torch,_torchlight)': 432, 'towel': 433, 'toy_(doll,_toy)': 434, 'tractor': 435, 'trash_(debris,_garbage,_litter,_trash,_waste)': 436, 'tray': 437, 'treadmill': 438, 'tree': 439, 'trimmer_(pruner,_trimmer)': 440, 'trowel': 441, 'truck': 442, 'tweezer': 443, 'umbrella': 444, 'undergarment_(boxer,_bra)': 445, 'vacuum': 446, 'vacuum_cleaner': 447, 'valve': 448, 'vase': 449, 'video_game': 450, 'violin': 451, 'wall': 452, 'wallet': 453, 'wallpaper': 454, 'washing_machine': 455, 'watch': 456, 'water': 457, 'watermelon': 458, 'weighing_scale': 459, 'welding_torch': 460, 'wheat_(maize,_wheat)': 461, 'wheel_(tyre,_wheel)': 462, 'wheelbarrow': 463, 'whisk': 464, 'window': 465, 'windshield': 466, 'wiper_(car)': 467, 'wire_(adapter,_cable,_charger,_connector,_cord,_wire)': 468, 'wood_(fiber,_firewood,_floorboard,_log,_lumber,_plank,_plywood,_timber,_wood,_woodcraft,_woodwork)': 469, 'worm': 470, 'wrapper_(covering,_film,_seal,_wrap,_wrapper,_wrapping)': 471, 'wrench_(spanner,_wrench)': 472, 'yam': 473, 'yeast': 474, 'yoghurt': 475, 'zipper_(zip,_zipper)': 476, 'zucchini': 477, 'ambulance': 478, 'back': 479, 'bamboo': 480, 'bandage': 481, 'baton': 482, 'bird': 483, 'brownie': 484, 'cake': 485, 'cash_register': 486, 'cassava': 487, 'cocoa': 488, 'courgette': 489, 'cow': 490, 'cupcake': 491, 'drone': 492, 'earplug': 493, 'hotdog': 494, 'juicer': 495, 'kiwi': 496, 'ladle': 497, 'leek': 498, 'lettuce': 499, 'marble': 500, 'melon': 501, 'orange': 502, 'peach': 503, 'person_(herself,_himself,_lady,_man,_person,_shoulder,_they,_woman)': 504, 'pipette': 505, 'plum': 506, 'plunger': 507, 'printer': 508, 'putty': 509, 'racket': 510, 'ratchet': 511, 'road': 512, 'salad': 513, 'scaffold': 514, 'squash': 515, 'stereo': 516, 'strawberry': 517, 'thermometer': 518, 'transistor': 519, 'vinegar': 520},
),
'vocab_verb': Vocabulary(
  counter=Counter({'adjust_(regulate,_increase/reduce,_change)': 1, 'apply_(spread,_smear)': 1, 'arrange_(straighten,_sort,_distribute,_align)': 1, 'attach_(plug-in,_join,_fasten,_connect,_attach)': 1, 'blow': 1, 'break': 1, 'carry': 1, 'catch': 1, 'clap': 1, 'clean_(sweep,_scrub,_mop,_dust)': 1, 'climb': 1, 'close': 1, 'consume_(taste,_sip,_eat,_drink)': 1, 'count': 1, 'cover': 1, 'crochet': 1, 'cut_(trim,_slice,_chop)': 1, 'detach_(unplug,_unhook,_disconnect)': 1, 'dig': 1, 'dip': 1, 'divide_(split,_separate)': 1, 'draw': 1, 'drill': 1, 'drive_(ride,_drive)': 1, 'enter': 1, 'feed': 1, 'file_(with_tool)': 1, 'fill': 1, 'fold': 1, 'fry': 1, 'give': 1, 'grate': 1, 'grind': 1, 'hang': 1, 'hit_(knock,_hit,_hammer)': 1, 'hold_(support,_grip,_grasp)': 1, 'insert': 1, 'inspect_(check,_look,_examine,_view)': 1, 'iron': 1, 'kick': 1, 'knead': 1, 'knit': 1, 'lift': 1, 'lock': 1, 'loosen': 1, 'mark': 1, 'measure_(weigh,_measure)': 1, 'mix': 1, 'mold': 1, 'move_(transfer,_pass,_exchange)': 1, 'open': 1, 'operate_(use,_dial,_click-button)': 1, 'pack': 1, 'paint': 1, 'park': 1, 'peel': 1, 'pet': 1, 'plant': 1, 'play': 1, 'point': 1, 'pour': 1, 'press': 1, 'pull': 1, 'pump': 1, 'push': 1, 'put_(place,_leave,_drop)': 1, 'read': 1, 'remove': 1, 'repair': 1, 'roll': 1, 'sand': 1, 'scoop': 1, 'scrape': 1, 'screw': 1, 'scroll': 1, 'search': 1, 'serve': 1, 'sew_(weave,_stitch,_sew)': 1, 'shake': 1, 'sharpen': 1, 'shuffle': 1, 'sieve': 1, 'sit': 1, 'smooth': 1, 'spray': 1, 'sprinkle': 1, 'squeeze': 1, 'stand': 1, 'step': 1, 'stick_(tape,_stick,_glue)': 1, 'stretch': 1, 'swing': 1, 'take_(pick,_grab,_get)': 1, 'talk_(talk,_interact,_converse)': 1, 'throw_(toss,_dump,_dispose)': 1, 'tie': 1, 'tighten': 1, 'tilt': 1, 'touch': 1, 'turn_(spin,_rotate,_flip,_turn_over)': 1, 'turn_off_(turn_off,_switch_off)': 1, 'turn_on_(switch_on,_start,_light)': 1, 'uncover': 1, 'unfold': 1, 'unroll': 1, 'unscrew': 1, 'untie': 1, 'walk': 1, 'wash': 1, 'water': 1, 'wear': 1, 'weld': 1, 'wipe': 1, 'write': 1, 'zip': 1, 'watch': 1, 'wave': 1}),
  index_to_token=('adjust_(regulate,_increase/reduce,_change)', 'apply_(spread,_smear)', 'arrange_(straighten,_sort,_distribute,_align)', 'attach_(plug-in,_join,_fasten,_connect,_attach)', 'blow', 'break', 'carry', 'catch', 'clap', 'clean_(sweep,_scrub,_mop,_dust)', 'climb', 'close', 'consume_(taste,_sip,_eat,_drink)', 'count', 'cover', 'crochet', 'cut_(trim,_slice,_chop)', 'detach_(unplug,_unhook,_disconnect)', 'dig', 'dip', 'divide_(split,_separate)', 'draw', 'drill', 'drive_(ride,_drive)', 'enter', 'feed', 'file_(with_tool)', 'fill', 'fold', 'fry', 'give', 'grate', 'grind', 'hang', 'hit_(knock,_hit,_hammer)', 'hold_(support,_grip,_grasp)', 'insert', 'inspect_(check,_look,_examine,_view)', 'iron', 'kick', 'knead', 'knit', 'lift', 'lock', 'loosen', 'mark', 'measure_(weigh,_measure)', 'mix', 'mold', 'move_(transfer,_pass,_exchange)', 'open', 'operate_(use,_dial,_click-button)', 'pack', 'paint', 'park', 'peel', 'pet', 'plant', 'play', 'point', 'pour', 'press', 'pull', 'pump', 'push', 'put_(place,_leave,_drop)', 'read', 'remove', 'repair', 'roll', 'sand', 'scoop', 'scrape', 'screw', 'scroll', 'search', 'serve', 'sew_(weave,_stitch,_sew)', 'shake', 'sharpen', 'shuffle', 'sieve', 'sit', 'smooth', 'spray', 'sprinkle', 'squeeze', 'stand', 'step', 'stick_(tape,_stick,_glue)', 'stretch', 'swing', 'take_(pick,_grab,_get)', 'talk_(talk,_interact,_converse)', 'throw_(toss,_dump,_dispose)', 'tie', 'tighten', 'tilt', 'touch', 'turn_(spin,_rotate,_flip,_turn_over)', 'turn_off_(turn_off,_switch_off)', 'turn_on_(switch_on,_start,_light)', 'uncover', 'unfold', 'unroll', 'unscrew', 'untie', 'walk', 'wash', 'water', 'wear', 'weld', 'wipe', 'write', 'zip', 'watch', 'wave'),
  token_to_index={'adjust_(regulate,_increase/reduce,_change)': 0, 'apply_(spread,_smear)': 1, 'arrange_(straighten,_sort,_distribute,_align)': 2, 'attach_(plug-in,_join,_fasten,_connect,_attach)': 3, 'blow': 4, 'break': 5, 'carry': 6, 'catch': 7, 'clap': 8, 'clean_(sweep,_scrub,_mop,_dust)': 9, 'climb': 10, 'close': 11, 'consume_(taste,_sip,_eat,_drink)': 12, 'count': 13, 'cover': 14, 'crochet': 15, 'cut_(trim,_slice,_chop)': 16, 'detach_(unplug,_unhook,_disconnect)': 17, 'dig': 18, 'dip': 19, 'divide_(split,_separate)': 20, 'draw': 21, 'drill': 22, 'drive_(ride,_drive)': 23, 'enter': 24, 'feed': 25, 'file_(with_tool)': 26, 'fill': 27, 'fold': 28, 'fry': 29, 'give': 30, 'grate': 31, 'grind': 32, 'hang': 33, 'hit_(knock,_hit,_hammer)': 34, 'hold_(support,_grip,_grasp)': 35, 'insert': 36, 'inspect_(check,_look,_examine,_view)': 37, 'iron': 38, 'kick': 39, 'knead': 40, 'knit': 41, 'lift': 42, 'lock': 43, 'loosen': 44, 'mark': 45, 'measure_(weigh,_measure)': 46, 'mix': 47, 'mold': 48, 'move_(transfer,_pass,_exchange)': 49, 'open': 50, 'operate_(use,_dial,_click-button)': 51, 'pack': 52, 'paint': 53, 'park': 54, 'peel': 55, 'pet': 56, 'plant': 57, 'play': 58, 'point': 59, 'pour': 60, 'press': 61, 'pull': 62, 'pump': 63, 'push': 64, 'put_(place,_leave,_drop)': 65, 'read': 66, 'remove': 67, 'repair': 68, 'roll': 69, 'sand': 70, 'scoop': 71, 'scrape': 72, 'screw': 73, 'scroll': 74, 'search': 75, 'serve': 76, 'sew_(weave,_stitch,_sew)': 77, 'shake': 78, 'sharpen': 79, 'shuffle': 80, 'sieve': 81, 'sit': 82, 'smooth': 83, 'spray': 84, 'sprinkle': 85, 'squeeze': 86, 'stand': 87, 'step': 88, 'stick_(tape,_stick,_glue)': 89, 'stretch': 90, 'swing': 91, 'take_(pick,_grab,_get)': 92, 'talk_(talk,_interact,_converse)': 93, 'throw_(toss,_dump,_dispose)': 94, 'tie': 95, 'tighten': 96, 'tilt': 97, 'touch': 98, 'turn_(spin,_rotate,_flip,_turn_over)': 99, 'turn_off_(turn_off,_switch_off)': 100, 'turn_on_(switch_on,_start,_light)': 101, 'uncover': 102, 'unfold': 103, 'unroll': 104, 'unscrew': 105, 'untie': 106, 'walk': 107, 'wash': 108, 'water': 109, 'wear': 110, 'weld': 111, 'wipe': 112, 'write': 113, 'zip': 114, 'watch': 115, 'wave': 116},
)}
```

`arctix` provides the function `arctix.dataset.ego4d.to_array` to convert
the `polars.DataFrame` to a dictionary of numpy arrays.

```pycon

>>> from arctix.dataset.ego4d import to_array
>>> arrays = to_array(data)  # doctest: +SKIP

```

The dictionary contains some regular arrays and masked arrays because sequences have variable
lengths:

```textmate
{'noun': masked_array(
  data=[['container_(box,_can,_carton,_case,_casing,_container,_crate,_holder,_jar,_jerrycan,_keg,_pack,_package,_packaging,_packet,_storage,_tank,_tin)',
         'container_(box,_can,_carton,_case,_casing,_container,_crate,_holder,_jar,_jerrycan,_keg,_pack,_package,_packaging,_packet,_storage,_tank,_tin)',
         'scissors', ..., --, --, --],
        ['paintbrush', 'paintbrush', 'brush', ..., --, --, --],
        ['faucet_(faucet,_tap)',
         'napkin_(handkerchief,_napkin,_serviette,_tissue,_wipe)',
         'napkin_(handkerchief,_napkin,_serviette,_tissue,_wipe)', ...,
         --, --, --],
        ...,
        ['metal_(lead,_metal,_steel)', 'metal_(lead,_metal,_steel)',
         'metal_(lead,_metal,_steel)', ..., --, --, --],
        ['paper_(chart,_craft,_newspaper,_note,_paper,_papercraft,_poster,_receipt)',
         'door', 'switch_(knob,_switch)', ..., --, --, --],
        ['hose', 'hose', 'spray_(spray,_sprayer)', ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value='N/A',
  dtype='<U142'), 'noun_label': masked_array(
  data=[[105, 105, 349, ..., --, --, --],
        [279, 279, 50, ..., --, --, --],
        [150, 265, 265, ..., --, --, --],
        ...,
        [247, 247, 247, ..., --, --, --],
        [286, 129, 409, ..., --, --, --],
        [211, 211, 391, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=999999), 'split': array(['train', 'train', 'train', ..., 'train', 'train', 'train'],
      dtype='<U5'), 'sequence_length': array([73, 31, 68, ..., 64, 22,  3]), 'action_clip_start_frame': masked_array(
  data=[[0, 8, 25, ..., --, --, --],
        [137, 363, 655, ..., --, --, --],
        [144, 210, 198, ..., --, --, --],
        ...,
        [155, 283, 299, ..., --, --, --],
        [422, 1003, 1201, ..., --, --, --],
        [1673, 1681, 2740, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=999999), 'action_clip_start_sec': masked_array(
  data=[[0.021028645833333335, 0.2876953125, 0.8543619791666667, ..., --,
         --, --],
        [4.554361979166686, 12.087695266666685, 21.821028599999977, ...,
         --, --, --],
        [4.787695266666674, 6.9876952666666625, 6.587695266666685, ...,
         --, --, --],
        ...,
        [5.151028600000018, 9.41769526666667, 9.951028599999972, ..., --,
         --, --],
        [14.066666666666606, 33.433333333333394, 40.0333333333333, ...,
         --, --, --],
        [55.75436193333337, 56.02102860000002, 91.32102860000009, ...,
         --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=1e+20), 'action_clip_end_frame': masked_array(
  data=[[194, 248, 265, ..., --, --, --],
        [377, 603, 895, ..., --, --, --],
        [384, 450, 438, ..., --, --, --],
        ...,
        [395, 523, 539, ..., --, --, --],
        [662, 1243, 1441, ..., --, --, --],
        [1913, 1921, 2980, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=999999), 'action_clip_end_sec': masked_array(
  data=[[6.4876953125, 8.2876953125, 8.854361979166667, ..., --, --, --],
        [12.554361979166686, 20.087695266666685, 29.821028599999977, ...,
         --, --, --],
        [12.787695266666674, 14.987695266666663, 14.587695266666685, ...,
         --, --, --],
        ...,
        [13.151028600000018, 17.41769526666667, 17.951028599999972, ...,
         --, --, --],
        [22.066666666666606, 41.433333333333394, 48.0333333333333, ...,
         --, --, --],
        [63.75436193333337, 64.02102860000002, 99.32102860000009, ...,
         --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=1e+20), 'verb': masked_array(
  data=[['take_(pick,_grab,_get)', 'put_(place,_leave,_drop)',
         'take_(pick,_grab,_get)', ..., --, --, --],
        ['remove', 'dip', 'dip', ..., --, --, --],
        ['turn_off_(turn_off,_switch_off)',
         'clean_(sweep,_scrub,_mop,_dust)', 'take_(pick,_grab,_get)',
         ..., --, --, --],
        ...,
        ['remove', 'remove', 'put_(place,_leave,_drop)', ..., --, --, --],
        ['take_(pick,_grab,_get)', 'open',
         'turn_on_(switch_on,_start,_light)', ..., --, --, --],
        ['turn_on_(switch_on,_start,_light)', 'spray',
         'turn_off_(turn_off,_switch_off)', ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value='N/A',
  dtype='<U47'), 'verb_label': masked_array(
  data=[[92, 65, 92, ..., --, --, --],
        [67, 19, 19, ..., --, --, --],
        [100, 9, 92, ..., --, --, --],
        ...,
        [67, 67, 65, ..., --, --, --],
        [92, 50, 101, ..., --, --, --],
        [101, 84, 100, ..., --, --, --]],
  mask=[[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True]],
  fill_value=999999), 'clip_uid': array(['002e11bc-deef-45f7-9af8-59421a606d69',
       '00600a90-0934-4d5f-aa9d-1cdea46ab740',
       '00be9fe7-617c-46cf-a0de-7432896c1705', ...,
       'ffab3f83-5c60-431c-8d5d-543695fd11f7',
       'ffdbd08d-d9e0-4669-b043-6f3d1c02d4d8',
       'ffffec25-661a-4450-ad99-6f3d58144930'], dtype='<U36')}
```
