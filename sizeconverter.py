def convert_to_us_ring_size(mm: float) -> str:
    # Diameter-to-US size mapping from the chart
    size_chart = {
        14.1: 3, 14.5: 3.5, 14.9: 4,
        15.3: 4.5, 15.7: 5,
        16.1: 5.5, 16.5: 6, 16.9: 6.5,
        17.3: 7, 17.7: 7.5,
        18.1: 8, 18.5: 8.5,
        19.0: 9, 19.4: 9.5, 19.8: 10,
        20.2: 10.5, 20.6: 11,
        21.0: 11.5, 21.4: 12, 21.8: 12.5,
        22.2: 13, 22.6: 13.5
    }

    # Sort known diameters
    sorted_sizes = sorted(size_chart.keys())

    # Exact match
    if mm in size_chart:
        return str(size_chart[mm]), f'{size_chart[mm]}'

    # Find the nearest bounds for interpolation
    lower = max((d for d in sorted_sizes if d <= mm), default=None)
    upper = min((d for d in sorted_sizes if d >= mm), default=None)

    if lower is None or upper is None:
        return 'invalid', 'invalid'

    # Linear interpolation between known sizes
    lower_size = size_chart[lower]
    upper_size = size_chart[upper]

    if lower == upper:
        return str(lower_size)

    # Interpolate proportionally
    interpolated_size = lower_size + (upper_size - lower_size) * ((mm - lower) / (upper - lower))
    rounded_size = round(interpolated_size * 2) / 2
    #print(f'closest size {lower_size} and {upper_size}')
    best_match = f'{lower_size} & {upper_size}'
    return rounded_size, best_match

def convert_all_ring_sizes(measurements: dict) -> dict:
    # Convert each finger measurement to US size
    converted_sizes = {}
    for finger, size in measurements.items():
        us_size, best_match = convert_to_us_ring_size(float(size))
        converted_sizes[finger] = {
            'mm': float(size),
            'US': us_size,
            'bestMatch': best_match,
        }
    return converted_sizes

