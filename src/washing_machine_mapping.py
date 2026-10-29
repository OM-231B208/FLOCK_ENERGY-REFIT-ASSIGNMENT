"""Official cleaned REFIT readme mapping: household to washing-machine IAM."""

WASHING_MACHINE_CHANNELS: dict[int, tuple[str, ...]] = {
    1: ("Appliance5",),
    2: ("Appliance2",),
    3: ("Appliance6",),
    4: ("Appliance4", "Appliance5"),
    5: ("Appliance3",),
    6: ("Appliance2",),
    7: ("Appliance5",),
    8: ("Appliance4",),
    9: ("Appliance3",),
    10: ("Appliance5",),
    11: ("Appliance3",),
    13: ("Appliance3",),
    15: ("Appliance3",),
    16: ("Appliance5",),
    17: ("Appliance4",),
    18: ("Appliance5",),
    19: ("Appliance2",),
    20: ("Appliance4",),
    21: ("Appliance3",),
}

SELECTED_HOUSES = (1, 2, 3, 5)
MODEL_TRAIN_HOUSES = (1, 2, 3)
MODEL_TEST_HOUSE = 5
