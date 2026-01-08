from obspy import read, read_inventory
import os
from pathlib import Path
from obspy.signal import PPSD
import sys
import numpy as np
from datetime import datetime, time as dt_time, timedelta
from obspy import UTCDateTime


def find_miniseed_channels(folder):
    extensions = [".mseed", ".msd", ".miniseed"]
    seen = set()
    for ext in extensions:
        for path in Path(folder).rglob(f"*{ext}"):
            try:
                st = read(str(path))
                for tr in st:
                    if len(tr) == 0:
                        continue
                    key = (tr.stats.location.strip(), tr.stats.channel.strip())
                    if key not in seen:
                        seen.add(key)
                if seen:
                    return sorted(seen)
            except Exception:
                continue
    return []


def find_miniseed(workdir, channel, location=None):
    for file in Path(workdir).rglob("*"):
        if file.suffix.lower() in [".msd", ".miniseed", ".mseed"]:
            try:
                st = read(str(file))
                for tr in st:
                    if location:
                        if (
                            tr.stats.channel == channel
                            and tr.stats.location == location
                        ):
                            return str(file)
                    else:
                        if tr.stats.channel == channel:
                            return str(file)
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
    return None


def parse_npz_timestamp(filename):
    """
    Parse timestamp from npz filename.
    Format: yy-mm-dd_HH-MM-SS.ffffff.npz
    Returns datetime object or None if parsing fails.
    """
    try:
        stem = Path(filename).stem  # Remove .npz extension
        # Parse the timestamp: yy-mm-dd_HH-MM-SS.ffffff
        dt = datetime.strptime(stem, '%y-%m-%d_%H-%M-%S.%f')
        return dt
    except Exception:
        return None


def is_time_in_range(dt, start_time, end_time):
    """
    Check if datetime's time component falls within start_time and end_time.
    Handles ranges that span midnight (e.g., 22:00 to 06:00).

    Args:
        dt: datetime object
        start_time: time object (e.g., time(22, 0))
        end_time: time object (e.g., time(6, 0))

    Returns:
        True if dt.time() is within the range, False otherwise
    """
    if start_time is None or end_time is None:
        return True

    t = dt.time()

    # Normal range (e.g., 06:00 to 22:00)
    if start_time <= end_time:
        return start_time <= t <= end_time
    # Range spans midnight (e.g., 22:00 to 06:00)
    else:
        return t >= start_time or t <= end_time


def filter_npz_files_by_time(npz_files, time_filter):
    """
    Filter npz files based on time_filter configuration.
    
    With the new approach, files are split during calculation and have
    _day or _night suffix based on the original time_filter used during calculation.
    This function filters files to match the requested time range.

    Args:
        npz_files: list of Path objects
        time_filter: dict with optional 'night_start' and 'night_stop' keys
                    (e.g., {'night_start': '22:00', 'night_stop': '06:00'})
                    These specify the desired time range to filter for.

    Returns:
        filtered list of Path objects
    """
    if not time_filter:
        return npz_files

    night_start_str = time_filter.get('night_start')
    night_stop_str = time_filter.get('night_stop')

    if not night_start_str or not night_stop_str:
        return npz_files

    # Parse time filter
    try:
        time_formats = ['%H:%M', '%H:%M:%S']
        filter_start = None
        filter_stop = None
        
        for fmt in time_formats:
            try:
                filter_start = datetime.strptime(night_start_str, fmt).time()
                filter_stop = datetime.strptime(night_stop_str, fmt).time()
                break
            except ValueError:
                continue
        
        if filter_start is None or filter_stop is None:
            print(f"Warning: Invalid time format in time_filter. "
                  f"Expected 'HH:MM' or 'HH:MM:SS', got start='{night_start_str}', "
                  f"stop='{night_stop_str}'. Using all files.")
            return npz_files
    except Exception:
        return npz_files
    
    # Determine if the time range spans midnight (night) or not (day)
    # If filter_start > filter_stop (e.g., 22:00 to 06:00), it's a night range
    # If filter_start <= filter_stop (e.g., 06:00 to 22:00), it's a day range
    is_night_range = filter_start > filter_stop
    
    filtered = []
    for file in npz_files:
        filename = file.stem  # Get filename without .npz
        
        # Check if this is a labeled file (new format)
        if filename.endswith('_night'):
            # This file contains night data
            # Include if we're filtering for night range
            if is_night_range:
                filtered.append(file)
        elif filename.endswith('_day'):
            # This file contains day data
            # Include if we're filtering for day range
            if not is_night_range:
                filtered.append(file)
        else:
            # Old format without label - use timestamp-based filtering
            # parse_npz_timestamp expects the full filename with .npz
            dt = parse_npz_timestamp(file.name)
            if dt and is_time_in_range(dt, filter_start, filter_stop):
                filtered.append(file)

    return filtered


def split_trace_by_time_filter(trace, time_filter):
    """
    Split a trace into segments based on time_filter.
    
    Args:
        trace: ObsPy Trace object
        time_filter: dict with 'night_start' and 'night_stop' keys (HH:MM format)
                    or None to return trace as-is
    
    Returns:
        list of tuples: [(trace_segment, label), ...]
        where label is 'day' or 'night' or 'all' (if no filter)
    """
    if not time_filter:
        return [(trace, 'all')]
    
    night_start_str = time_filter.get('night_start')
    night_stop_str = time_filter.get('night_stop')
    
    if not night_start_str or not night_stop_str:
        return [(trace, 'all')]
    
    try:
        # Parse time strings
        time_formats = ['%H:%M', '%H:%M:%S']
        night_start = None
        night_stop = None
        
        for fmt in time_formats:
            try:
                night_start = datetime.strptime(night_start_str, fmt).time()
                night_stop = datetime.strptime(night_stop_str, fmt).time()
                break
            except ValueError:
                continue
        
        if night_start is None or night_stop is None:
            return [(trace, 'all')]
        
    except Exception:
        return [(trace, 'all')]
    
    # Get trace time range
    starttime = trace.stats.starttime
    endtime = trace.stats.endtime
    
    # Check if entire trace falls within one period
    start_dt = starttime.datetime
    end_dt = endtime.datetime
    
    # If trace is within a single period (day or night), no splitting needed
    start_in_range = is_time_in_range(start_dt, night_start, night_stop)
    end_in_range = is_time_in_range(end_dt, night_start, night_stop)
    
    # If both start and end are in the same period and trace is short enough
    # (less than 12 hours to avoid spanning multiple days)
    trace_duration_hours = (endtime - starttime) / 3600.0
    if start_in_range == end_in_range and trace_duration_hours < 12:
        label = 'night' if start_in_range else 'day'
        return [(trace, label)]
    
    # Need to split the trace - find all transition times
    result = []
    current_time = UTCDateTime(starttime)
    
    # Iterate through each day in the trace
    while current_time < endtime:
        current_date = current_time.datetime.date()
        
        # Calculate transition times for this day
        # Night start time on current day
        night_start_utc = UTCDateTime(
            datetime.combine(current_date, night_start)
        )
        
        # Determine night end time
        if night_start <= night_stop:
            # Normal range (e.g., 06:00 to 22:00 is day)
            # So night is outside this range
            # This means we have TWO night periods in a day
            # This case is complex, let's handle the common case first
            night_end_utc = UTCDateTime(
                datetime.combine(current_date, night_stop)
            )
        else:
            # Night spans midnight (e.g., 22:00 to 06:00)
            # Night ends next day
            next_date = current_date + timedelta(days=1)
            night_end_utc = UTCDateTime(
                datetime.combine(next_date, night_stop)
            )
        
        # Create segments for this day
        # We'll just split at the boundaries and label each segment
        day_boundaries = []
        
        if night_start <= night_stop:
            # Day period is FROM night_stop TO night_start
            # (e.g., 06:00-22:00 is day, rest is night)
            day_start = UTCDateTime(datetime.combine(current_date, night_stop))
            day_end = UTCDateTime(datetime.combine(current_date, night_start))
            
            # Night before day_start
            if current_time < day_start and day_start < endtime:
                seg_end = min(day_start, endtime)
                if seg_end > current_time:
                    day_boundaries.append((current_time, seg_end, 'night'))
                current_time = seg_end
            
            # Day period
            if current_time < day_end and day_end <= endtime:
                seg_end = min(day_end, endtime)
                if seg_end > current_time:
                    day_boundaries.append((current_time, seg_end, 'day'))
                current_time = seg_end
            
            # Night after day_end
            next_day_start = UTCDateTime(
                datetime.combine(current_date + timedelta(days=1), dt_time(0, 0))
            )
            seg_end = min(next_day_start, endtime)
            if seg_end > current_time:
                day_boundaries.append((current_time, seg_end, 'night'))
            
        else:
            # Night spans midnight (common case)
            # Night is FROM night_start TO night_stop (next day)
            # Day is FROM night_stop TO night_start
            day_start = night_end_utc  # night_stop of current day
            day_end = night_start_utc   # night_start of current day
            
            # If we're in the night period from previous day
            if current_time < day_start and day_start <= endtime:
                seg_end = min(day_start, endtime)
                if seg_end > current_time:
                    day_boundaries.append((current_time, seg_end, 'night'))
                current_time = seg_end
            
            # Day period
            if current_time < day_end and day_end <= endtime:
                seg_end = min(day_end, endtime)
                if seg_end > current_time:
                    day_boundaries.append((current_time, seg_end, 'day'))
                current_time = seg_end
            
            # Night period starting at night_start
            next_day_start = UTCDateTime(
                datetime.combine(current_date + timedelta(days=1), dt_time(0, 0))
            )
            # Night continues until midnight, then we process next day
            seg_end = min(next_day_start, endtime)
            if seg_end > current_time:
                day_boundaries.append((current_time, seg_end, 'night'))
        
        # Move to next day
        current_time = UTCDateTime(
            datetime.combine(current_date + timedelta(days=1), dt_time(0, 0))
        )
    
    # Create trace segments
    for seg_start, seg_end, label in day_boundaries:
        if seg_end > seg_start:
            try:
                seg_trace = trace.slice(seg_start, seg_end)
                if len(seg_trace.data) > 0:
                    result.append((seg_trace, label))
            except Exception as e:
                print(f"Warning: Could not slice trace: {e}")
                continue
    
    # If we couldn't split for some reason, return original
    if not result:
        return [(trace, 'all')]
    
    return result


def calculate_ppsd_worker(job_list, inv_path, tw, folder, time_filter=None):
    inv = load_inventory(inv_path)

    for file, loc, chan in job_list:
        try:
            st = read(str(file))
            st = st.select(channel=chan, location=loc if loc else "")
            if not st:
                continue
        except Exception as e:
            print(f"Read error in {file.name}: {e}")
            continue

        for tr in st:
            try:
                if loc:
                    npzfolder = (
                        Path(resource_path(folder)) / f"npz_{loc}_{chan}"
                    )
                else:
                    npzfolder = Path(resource_path(folder)) / f"npz_{chan}"

                npzfolder.mkdir(exist_ok=True)
                
                # Split trace by time filter if needed
                trace_segments = split_trace_by_time_filter(tr, time_filter)
                
                for trace_segment, label in trace_segments:
                    ppsd = PPSD(trace_segment.stats, metadata=inv, ppsd_length=tw)
                    ppsd.add(trace_segment)
                    timestamp = trace_segment.stats.starttime.strftime("%y-%m-%d_%H-%M-%S.%f")
                    
                    # Include label in filename if filtering is applied
                    if label != 'all':
                        outfile = npzfolder / f"{timestamp}_{label}.npz"
                    else:
                        outfile = npzfolder / f"{timestamp}.npz"
                    
                    ppsd.save_npz(str(outfile))
            except Exception as e:
                print(
                    f"[{os.getpid()}] Error processing {file.name}"
                    f" trace {tr.id}: {e}"
                    )


def load_inventory(resp_file):
    ext = Path(resp_file).suffix.lower()

    if ext in [".seed", ".dataless"]:
        fmt = "SEED"
    elif ext == ".xml":
        fmt = "STATIONXML"
    else:
        fmt = None

    try:
        if fmt:
            inv = read_inventory(resp_file, format=fmt)
        else:
            inv = read_inventory(resp_file)
        return inv
    except Exception as e:
        print(f"Failed to read inventory {resp_file}: {e}")
        return


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_path, relative_path))


def calculate_noise_line(amplitude_mg, freq_range=(1.0, 10.0)):

    A_rms = amplitude_mg * 9.80665e-3
    f1, f2 = freq_range
    bandwidth = abs(f2 - f1)
    if bandwidth <= 0:
        raise ValueError("Frequency range must have non-zero width")

    PSD = (A_rms ** 2) / bandwidth
    dB_value = 10 * np.log10(PSD)
    return f1, f2, dB_value
