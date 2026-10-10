"""Groundnut stage advice from TNAU Agritech guidance.

The ML model does not know the crop. This layer adds timed tasks and stage
advice on top of the model's risk level.
Sources: TNAU groundnut guide (irrigated and rainfed) and TNAU drought notes.
"""
import pandas as pd

# Pod duration in days, from the TNAU variety table
VARIETY_DAYS = {
    'TMV 7': (100, 105), 'TMV 10': (120, 130), 'CO 3': (115, 120),
    'COGn 4': (115, 120), 'COGn 5': (125, 130), 'ALR 3': (110, 115),
    'VRI 2': (100, 105), 'VRI 3': (90, 90), 'VRIGn 5': (105, 110),
}

# (from_day, to_day, task, mode); days are after sowing (DAS)
# mode: 'irrigated', 'rainfed' or 'both'
TASKS = [
    (4, 5, 'Life irrigation to break the surface crust (only if sowing irrigation was given).', 'irrigated'),
    (20, 20, 'Irrigate at 20 days after sowing.', 'irrigated'),
    (20, 20, 'Hand weed and hoe, if no herbicide was used.', 'both'),
    (25, 25, 'Spray the combined nutrient solution (first of two sprays).', 'both'),
    (26, 60, 'At flowering, give two irrigations. Spray 0.5% potassium chloride to reduce water-stress damage.', 'irrigated'),
    (26, 60, 'At pegging, give one or two irrigations.', 'irrigated'),
    (35, 35, 'Spray the combined nutrient solution (second spray).', 'both'),
    (40, 40, 'Hand weed and hoe on day 40, if no herbicide was used. Then earth up.', 'both'),
    (40, 45, 'Apply gypsum at 400 kg/ha beside the plants, hoe it in, then earth up. Avoid gypsum in calcareous soils.', 'irrigated'),
    (40, 70, 'Apply gypsum at 400 kg/ha beside the plants, timed by soil moisture. Avoid gypsum in calcareous soils.', 'rainfed'),
    (45, 999, 'Do not disturb the soil after day 45, as it hurts pod formation.', 'both'),
    (61, 999, 'Pod development: give two to three irrigations, depending on soil type.', 'irrigated'),
]


def das(sown_on, on):
    return (pd.Timestamp(on) - pd.Timestamp(sown_on)).days


def farm_advice(farm, on, level):
    d = das(farm['sown_on'], on)
    mode = farm.get('irrigation', 'irrigated')
    if d < 0:
        return {'das': d, 'stage': 'Not yet sown', 'critical': False,
                'stage_text': 'The field is not sown on this date.', 'tasks': [], 'note': ''}

    lo, hi = VARIETY_DAYS.get(farm.get('variety', 'TMV 7'), (100, 105))
    if d <= 25:
        stage, critical = 'Early growth', False
        stage_text = 'Keep the soil moist and stop the surface from crusting.'
    elif d <= 60:
        stage, critical = 'Flowering and pegging', True
        stage_text = ('Critical stage. Pegging, flowering and pod development need steady soil '
                      'moisture. Dry spells now cut yield.')
    elif d <= hi:
        stage, critical = 'Pod development', True
        stage_text = ('Pods are filling. Keep moisture adequate. Bold-seeded varieties are hit '
                      'hardest when water runs short.')
    else:
        stage, critical = 'Harvest period', False
        stage_text = 'The crop is near maturity. Check maturity before harvest.'

    # Tasks due within the next 7 days
    tasks = [t for lo_d, hi_d, t, m in TASKS
             if (m == 'both' or m == mode) and lo_d <= d + 7 and hi_d >= d]

    if lo - 7 <= d <= hi + 7:
        tasks.append('Maturity check: pull a few plants and shell the pods. Ready when the inner '
                     'shell is brownish black, not white. Irrigate before harvest if the soil is dry.')

    note = ''
    if critical and level in ('Medium', 'High'):
        note = ('Critical stage with stress risk: irrigate within 3 days if no rain is forecast.'
                if level == 'High'
                else 'Critical stage: check soil moisture within 2 days.')

    return {'das': d, 'stage': stage, 'critical': critical,
            'stage_text': stage_text, 'tasks': tasks, 'note': note}