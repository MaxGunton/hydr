import os
import argparse
import math

import simplekml
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import utm
from shapely.geometry import Point, LineString
from matplotlib.cm import gist_rainbow
from hydr.utils import ok_to_write


MEAN_EARTH_RADIUS = 6371000  # meters
CENTRAL_MERIDIAN_SCALE_FACTOR = 0.9996


def get_midpoint(pt1, pt2) -> tuple:
    p1 = Point(pt1[0], pt1[1])
    p2 = Point(pt2[0], pt2[1])
    line = LineString([p1, p2])
    mp = line.interpolate(0.5, normalized=True)  # midpoint of linestring
    mp = list(mp.coords)[0]
    return mp


def distance(pt1, pt2) -> float:
    e1, n1 = pt1
    e2, n2 = pt2
    d = math.sqrt((e2 - e1) ** 2 + (n2 - n1) ** 2)
    return d


def ef_scale_factor(distance_above_sea_level):
    return MEAN_EARTH_RADIUS / (MEAN_EARTH_RADIUS + distance_above_sea_level)


def grid_to_ground_scale_factor(grid_distance, p1, p2, midpoint, cmz):
    k0 = CENTRAL_MERIDIAN_SCALE_FACTOR
    lambda0 = math.radians(cmz)

    # fi and lambda = WGS87 coordinates (lat, lon)
    p1_fi, p1_lambda = math.radians(p1[0]), math.radians(p1[1])
    p2_fi, p2_lambda = math.radians(p2[0]), math.radians(p2[1])
    mp_fi, mp_lambda = math.radians(midpoint[0]), math.radians(midpoint[1])

    kp1 = k0 * (1 + (p1_lambda - lambda0) ** 2 * math.cos(p1_fi) ** 2 / 2)
    kp2 = k0 * (1 + (p2_lambda - lambda0) ** 2 * math.cos(p2_fi) ** 2 / 2)
    kmp = k0 * (1 + (mp_lambda - lambda0) ** 2 * math.cos(mp_fi) ** 2 / 2)

    if grid_distance < 2000:
        # single scale factor
        return kp1
    elif 2000 <= grid_distance <= 4000:
        # average of both ends
        return (kp1 + kp2) / 2
    else:
        # simpsons rule (1 sixth of each end and two thirds from scale factor in middle)
        return (kp1 + 4 * kmp + kp2) / 6


def combined_scale_factor(grid_distance, distance_above_sea_level, p1, p2, midpoint, cmz):
    return grid_to_ground_scale_factor(grid_distance, p1, p2, midpoint, cmz) * ef_scale_factor(distance_above_sea_level)


def true_dist_between_lat_lon_pts(p1, p2):
    e1, n1, zn1, zl1 = utm.from_latlon(p1[0], p1[1])
    p1_grid = (e1, n1)

    e2, n2, zn2, zl2 = utm.from_latlon(p2[0], p2[1])
    p2_grid = (e2, n2)

    assert zn1 == zn2
    assert zl1 == zl2

    _, central_meridian_of_zone = utm.to_latlon(500000, n1, zn1, zl1)
    distance_grid = distance(p1_grid, p2_grid)
    midpoint_grid = get_midpoint(p1_grid, p2_grid)
    midpoint = utm.to_latlon(midpoint_grid[0], midpoint_grid[1], zn1, zl1)
    scale_factor = combined_scale_factor(distance_grid, 0, p1, p2, midpoint, central_meridian_of_zone)
    true_distance = distance_grid / scale_factor
    return true_distance


def draw_hull_perimeter(df, hull_indexes, kml):
    hull_indexes = hull_indexes.tolist()
    if len(hull_indexes) <= 1:
        return
    hull_indexes.append(hull_indexes[0])  # Add start to end to close the loop
    hull_style = simplekml.Style()
    hull_style.linestyle.color = 'ffb18ff4'
    hull_style.linestyle.width = 3  # 2
    for i in range(1, len(hull_indexes)):
        start_row = df.iloc[hull_indexes[i-1], :]
        end_row = df.iloc[hull_indexes[i], :]
        sn_0, lat_0, lon_0 = start_row['sn'], start_row['lat'], start_row['lon']
        sn_1, lat_1, lon_1 = end_row['sn'], end_row['lat'], end_row['lon']
        dist = '{:.3f}'.format(true_dist_between_lat_lon_pts((lat_0, lon_0), (lat_1, lon_1)))
        ls = kml.newlinestring(name=f'{sn_0} to {sn_1}', description=f'{dist} m',
                               coords=[(lon_0, lat_0), (lon_1, lat_1)])
        ls.style = hull_style


def draw_hull_area(df, hull_indexes, kml):
    hull_indexes = hull_indexes.tolist()
    if len(hull_indexes) <= 1:
        return
    hull_indexes.append(hull_indexes[0])  # add start to end to close the loop

    # set up hull area style
    hull_style = simplekml.Style()
    hull_style.polystyle.color = '77000000'
    hull_style.polystyle.outline = 0

    # get the coordinates in the correct form
    lat_idx, lon_idx = df.columns.get_loc('lat'), df.columns.get_loc('lon')
    coords = [(df.iloc[i, lon_idx], df.iloc[i, lat_idx]) for i in hull_indexes]

    # define the polygon and set the style
    poly = kml.newpolygon(name="Area of Interest", outerboundaryis=coords)
    poly.style = hull_style


def draw_hydrophones(row, kml):
    sn, lat, lon, lat_s, lon_s, color = row['sn'], row['lat'], row['lon'], row['lat_s'], row['lon_s'], row['color']
    name = f'SoundTrap (SN: {sn}) -- '
    descr = f'Latitude: {lat_s}\nLongitude: {lon_s}'
    hpt = kml.newpoint(name=name, description=descr, coords=[(lon, lat)])
    # hpt.style.labelstyle.scale = 2
    hpt.style.iconstyle.color = color
    # hpt.style.iconstyle.scale = 1
    # hpt.style.iconstyle.icon.href = 'https://cdn-icons-png.flaticon.com/512/2983/2983672.png'


def mpl_to_kml_color(color):
    color = [hex(int(c * 255)).split('x')[1] for c in color.tolist()]
    color.reverse()
    color = [f'0{c}' if len(c) == 1 else c for c in color]
    return ''.join(color)


def map_hydrophones(coord_file, dest):
    df = pd.read_csv(coord_file, dtype={'Serial Number': object})
    df = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'Serial Number': 'sn',
                            'Latitude (D\u00b0M\u02b9S.S")': 'lat_s', 'Longitude (D\u00b0M\u02b9S.S")': 'lon_s'})
    df = df.loc[:, ['sn', 'lat', 'lon', 'lat_s', 'lon_s']]
    h = df.shape[0] if df.shape[0] != 0 else 1
    df['color'] = np.apply_along_axis(mpl_to_kml_color, 1, gist_rainbow(np.arange(h)/h + 1/(2*h)))
    print(np.arange(h)/h + 1/(2*h))
    # df['color'] = mpl.colormaps['hsv'].resampled(df.shape[0])

    # 1) initialize kml
    kml = simplekml.Kml()
    kml.document.name = "Hydrophone Deployment"

    # 2) add the hydrophones as points
    df.apply(lambda x: draw_hydrophones(x, kml), axis=1)

    # 3) compute convex hull of the hydrophones (if 3 or more points)
    if df.shape[0] > 2:
        hpts = df[['lat', 'lon']].to_numpy()
        hull = ConvexHull(hpts)

        # 4) draw the hull perimeter
        draw_hull_perimeter(df, hull.vertices, kml)

        # 5) draw the hull area
        draw_hull_area(df, hull.vertices, kml)

    # 6) save our kml to disk
    dest = os.path.join(dest, 'deployment_map.kml')
    if ok_to_write(dest):
        kml.save(dest)


def total_to_dms(t):
    d = int(round(t // 3600))
    m = int(round((t - (d * 3600)) // 60))
    s = round(t - (d * 3600) - (m * 60), 8)
    return d, m, s


def clean_coord(deg, minute, sec):
    t = (deg * 3600) + (minute * 60) + sec
    d, m, s = total_to_dms(t)
    # assert (d * 3600) + (m * 60) + s == t
    print(f'{(d * 3600) + (m * 60) + s} == {t}')
    return d, m, s


def compute_full_coord(row):
    lat_deg, lat_min, lat_sec = clean_coord(row['lat_deg'], row['lat_min'], row['lat_sec'])
    latitude = (lat_deg + lat_min / 60 + lat_sec / 3600)
    latitude = -1 * latitude if row['south'] else latitude
    lat_deg = f'-{lat_deg}' if latitude < 0 else lat_deg

    lon_deg, lon_min, lon_sec = clean_coord(row['lon_deg'], row['lon_min'], row['lon_sec'])
    longitude = (lon_deg + lon_min / 60 + lon_sec / 3600)
    longitude = -1 * longitude if row['west'] else longitude
    lon_deg = f'-{lon_deg}' if longitude < 0 else lon_deg

    # return pd.Series((latitude, lat_deg, lat_min, lat_sec, longitude, lon_deg, lon_min, lon_sec))

    lat_dm = f'{lat_deg}\u00b0 {lat_min + lat_sec/60}\u02b9'
    lat_dms = f'{lat_deg}\u00b0 {lat_min}\u02b9 {lat_sec}"'

    lon_dm = f'{lon_deg}\u00b0 {lon_min + lon_sec / 60}\u02b9'
    lon_dms = f'{lon_deg}\u00b0 {lon_min}\u02b9 {lon_sec}"'

    return pd.Series((latitude, lat_dm, lat_dms, longitude, lon_dm, lon_dms))


def add_hydrophone_coords(dest):
    print("- When entering coordinates you can leave any sections blank and this will assume 0 for that component.  "
          "You can enter the entire coordinate under `Degrees` (or any for that matter) it will automatically split it "
          "up into the correct degrees, minutes and seconds (just leave the other fields blank, otherwise it will "
          "combine with that value (which is likely not intended).  \n"
          "- South is negative for `Latitude` and West is negative for `Longitude` and only the sign from the `Degree` "
          "component is used (others are ignored).  Therefore if for some reason your coordinates are entirely in "
          "Minutes, Seconds, or a combination of the two you can simply enter `-` into the Degree field to indicate "
          "that it is either S or W depending on the coordinate.  \n"
          "- If a hydrophone coordinate is missing simply leave it blank.  \n"
          "- When finished simple leave the Serial Number field blank and press enter.  ")
    sw_mappings = {'-': True, '': False}
    sns = []
    souths = []
    lat_degs = []
    lat_mins = []
    lat_secs = []
    wests = []
    lon_degs = []
    lon_mins = []
    lon_secs = []
    while True:
        sn = input("Serial Number (`Enter` to exit): ")
        # sn = 202118071
        if sn == '':
            break
        sns.append(sn)
        print("Latitude")
        print('--------')
        lad = input("Degrees: ")
        # lad = '-8'
        south = sw_mappings[lad] if lad in sw_mappings else None
        lad, south = (0.0, south) if south is not None else (abs(float(lad)), float(lad) < 0.0)
        lam = input("Minutes: ")
        # lam = '16.745'
        lam = abs(float(lam)) if lam != '' else 0.0
        las = input("Seconds: ")
        # las = ''
        las = abs(float(las)) if las != '' else 0.0
        souths.append(south)
        lat_degs.append(lad)
        lat_mins.append(lam)
        lat_secs.append(las)

        print("Longitude")
        print('--------')
        lod = input("Degrees: ")
        # lod = '116'
        west = sw_mappings[lod] if lod in sw_mappings else None
        lod, west = (0.0, west) if west is not None else (abs(float(lod)), float(lod) < 0.0)
        lom = input("Minutes: ")
        # lom = '41.615'
        lom = abs(float(lom)) if lom != '' else 0.0
        los = input("Seconds: ")
        # los = ''
        los = abs(float(los)) if los != '' else 0.0
        wests.append(west)
        lon_degs.append(lod)
        lon_mins.append(lom)
        lon_secs.append(los)

    df = pd.DataFrame({'Serial Number': sns, 'lat_deg': lat_degs, 'lat_min': lat_mins, 'lat_sec': lat_secs,
                       'lon_deg': lon_degs, 'lon_min': lon_mins, 'lon_sec': lon_secs, 'south': souths, 'west': wests})

    columns = ['Latitude', 'Latitude (D\u00b0M.M\u02b9)', 'Latitude (D\u00b0M\u02b9S.S")',
               'Longitude', 'Longitude (D\u00b0M.M\u02b9)', 'Longitude (D\u00b0M\u02b9S.S")']
    df[columns] = df.apply(compute_full_coord, axis=1)
    dest = os.path.join(dest, 'hydrophone_coordinates.csv')
    if ok_to_write(dest):
        df[['Serial Number'] + columns].to_csv(dest, index=False, encoding='utf8')


def add_hydrophone_coords_cli():
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dest', help='directory where to save hydrophone coord', default='.', type=str)
    args = parser.parse_args()

    # call the create_project_directory with the arguments passed
    add_hydrophone_coords(args.dest)


def map_hydrophones_cli():
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('coord', help='file containing hydrophone coordinates', default='.')
    parser.add_argument('-d', '--dest', help='directory where to save resulting kml', default='.')
    args = parser.parse_args()

    # call the create_project_directory with the arguments passed
    map_hydrophones(args.coord, args.dest)
