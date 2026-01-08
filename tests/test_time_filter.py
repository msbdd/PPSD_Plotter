import pytest
from pathlib import Path
from datetime import datetime, time
from src.ppsd_plotter_aux import (
    parse_npz_timestamp,
    is_time_in_range,
    filter_npz_files_by_time
)


def test_parse_npz_timestamp():
    """Test parsing timestamp from npz filename."""
    filename = '25-06-07_07-00-00.019536.npz'
    dt = parse_npz_timestamp(filename)
    assert dt is not None
    assert dt.year == 2025
    assert dt.month == 6
    assert dt.day == 7
    assert dt.hour == 7
    assert dt.minute == 0
    assert dt.second == 0


def test_parse_npz_timestamp_invalid():
    """Test parsing invalid filename returns None."""
    filename = 'invalid_filename.npz'
    dt = parse_npz_timestamp(filename)
    assert dt is None


def test_is_time_in_range_normal():
    """Test time range that doesn't span midnight."""
    dt = datetime(2025, 6, 7, 7, 0, 0)
    
    # Daytime range: 06:00 to 22:00
    day_start = time(6, 0)
    day_end = time(22, 0)
    
    assert is_time_in_range(dt, day_start, day_end) is True
    
    # Test time outside range
    dt_night = datetime(2025, 6, 7, 23, 0, 0)
    assert is_time_in_range(dt_night, day_start, day_end) is False


def test_is_time_in_range_spanning_midnight():
    """Test time range that spans midnight."""
    # Nighttime range: 22:00 to 06:00
    night_start = time(22, 0)
    night_end = time(6, 0)
    
    # Time at 23:00 should be in range
    dt_night = datetime(2025, 6, 7, 23, 0, 0)
    assert is_time_in_range(dt_night, night_start, night_end) is True
    
    # Time at 02:00 should be in range
    dt_early_morning = datetime(2025, 6, 7, 2, 0, 0)
    assert is_time_in_range(dt_early_morning, night_start, night_end) is True
    
    # Time at 07:00 should NOT be in range
    dt_day = datetime(2025, 6, 7, 7, 0, 0)
    assert is_time_in_range(dt_day, night_start, night_end) is False


def test_is_time_in_range_none_values():
    """Test that None values return True (no filtering)."""
    dt = datetime(2025, 6, 7, 7, 0, 0)
    assert is_time_in_range(dt, None, None) is True
    assert is_time_in_range(dt, None, time(22, 0)) is True
    assert is_time_in_range(dt, time(6, 0), None) is True


def test_filter_npz_files_by_time_no_filter():
    """Test that no filter returns all files."""
    files = [
        Path('25-06-07_07-00-00.019536.npz'),
        Path('25-06-07_23-00-00.019536.npz'),
    ]
    
    # No filter
    filtered = filter_npz_files_by_time(files, None)
    assert len(filtered) == 2
    
    # Empty filter
    filtered = filter_npz_files_by_time(files, {})
    assert len(filtered) == 2


def test_filter_npz_files_by_time_daytime():
    """Test filtering for daytime hours."""
    files = [
        Path('25-06-07_07-00-00.019536.npz'),  # 07:00 - day
        Path('25-06-07_23-00-00.019536.npz'),  # 23:00 - night
        Path('25-06-07_12-00-00.019536.npz'),  # 12:00 - day
        Path('25-06-07_02-00-00.019536.npz'),  # 02:00 - night
    ]
    
    time_filter = {'night_start': '06:00', 'night_stop': '22:00'}
    filtered = filter_npz_files_by_time(files, time_filter)
    
    assert len(filtered) == 2
    assert Path('25-06-07_07-00-00.019536.npz') in filtered
    assert Path('25-06-07_12-00-00.019536.npz') in filtered


def test_filter_npz_files_by_time_nighttime():
    """Test filtering for nighttime hours (spanning midnight)."""
    files = [
        Path('25-06-07_07-00-00.019536.npz'),  # 07:00 - day
        Path('25-06-07_23-00-00.019536.npz'),  # 23:00 - night
        Path('25-06-07_12-00-00.019536.npz'),  # 12:00 - day
        Path('25-06-07_02-00-00.019536.npz'),  # 02:00 - night
    ]
    
    time_filter = {'night_start': '22:00', 'night_stop': '06:00'}
    filtered = filter_npz_files_by_time(files, time_filter)
    
    assert len(filtered) == 2
    assert Path('25-06-07_23-00-00.019536.npz') in filtered
    assert Path('25-06-07_02-00-00.019536.npz') in filtered


def test_filter_npz_files_by_time_invalid_format():
    """Test that invalid time format returns all files."""
    files = [
        Path('25-06-07_07-00-00.019536.npz'),
        Path('25-06-07_23-00-00.019536.npz'),
    ]
    
    # Invalid time format
    time_filter = {'night_start': 'invalid', 'night_stop': '22:00'}
    filtered = filter_npz_files_by_time(files, time_filter)
    
    # Should return all files when format is invalid
    assert len(filtered) == 2


def test_filter_npz_files_by_time_missing_keys():
    """Test that missing keys return all files."""
    files = [
        Path('25-06-07_07-00-00.019536.npz'),
        Path('25-06-07_23-00-00.019536.npz'),
    ]
    
    # Missing night_stop
    time_filter = {'night_start': '22:00'}
    filtered = filter_npz_files_by_time(files, time_filter)
    assert len(filtered) == 2
    
    # Missing night_start
    time_filter = {'night_stop': '06:00'}
    filtered = filter_npz_files_by_time(files, time_filter)
    assert len(filtered) == 2
