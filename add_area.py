import sys
from geopy.geocoders import Nominatim
from pipeline import get_location

name = ' '.join(sys.argv[1:]).strip()
geo = Nominatim(user_agent='crop_stress_app').geocode(f'{name}, Tamil Nadu, India')
if geo is None:
    sys.exit('Place not found. Try a nearby town.')
get_location(name.title(), geo.latitude, geo.longitude)
print(f'Added {name.title()} ({geo.latitude:.4f}, {geo.longitude:.4f}). Now run export_dashboard.py')