from pipeline import LOCATIONS, get_location

for name, (lat, lon) in LOCATIONS.items():
    try:
        df = get_location(name, lat, lon)
        print(f'{name}: {len(df)} days, {df["target"].notna().sum()} labelled')
    except Exception as e:
        print(f'{name}: FAILED -> {e}')