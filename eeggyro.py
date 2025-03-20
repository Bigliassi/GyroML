import os
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import warnings

import mne
from scipy.signal import detrend
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skopt import gp_minimize
from skopt.space import Real, Categorical

import xgboost as xgb
from pykalman import KalmanFilter
import pywt

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# USER CONFIG
# -----------------------------------------------------------------------------
CHUNK_COUNT = 3                # Number of large time-based chunks
SUBCHUNK_DURATION = 5.0        # Sub-chunk size in seconds
SUBCHUNK_COUNT = 4             # How many sub-chunks per chunk
BAYES_CALLS = 8                # Bayesian optimization calls
RANDOM_SEED = 42

# We'll do ± 20% tolerance around the half-cycle time from the user’s nominal RPM
RPM_TOLERANCE = 0.2

# Confidence threshold for final pass classification
CONFIDENCE_THRESHOLD = 0.5

MAX_AXES_COMBO_SIZE = 2        # We'll test combos of up to 2 axes
MAX_SNIPPET_ATTEMPTS = 5
SNIPPET_SECONDS = 10.0

ALL_AXES = ['GyroX','GyroY','GyroZ','AccelerX','AccelerY','AccelerZ']

SEARCH_SPACE = [
    Real(1e-4, 0.02, name='initial_threshold', prior='log-uniform'),
    Real(0.5, 3.0, name='prominence_scale_factor', prior='log-uniform'),
    Categorical([3, 5, 8], name='cutoff_frequency'),
    Categorical([False, True], name='use_wavelet')
]

# -----------------------------------------------------------------------------
# PROMPT FOR FILENAME & RPM
# -----------------------------------------------------------------------------
def get_filename():
    """Prompt user for the BDF file name."""
    while True:
        fname = input("Enter the BDF filename (e.g., 'myfile.bdf'): ").strip()
        if os.path.isfile(fname):
            logging.info(f"File '{fname}' found.")
            return fname
        else:
            logging.error(f"File '{fname}' not found.")
            ans = input("Try again? (y/n): ").strip().lower()
            if ans != 'y':
                raise FileNotFoundError(f"File '{fname}' not found.")

def get_rpm():
    """Prompt user for the nominal RPM (cycles per minute)."""
    while True:
        val_str = input("Enter the nominal RPM (e.g. 60): ").strip()
        try:
            val = float(val_str)
            if val>0:
                return val
            else:
                print("Please enter a positive number for RPM.")
        except ValueError:
            print("Invalid input. Must be numeric, e.g. 60, 65.0, etc.")

def compute_rpm_range(rpm, tolerance=RPM_TOLERANCE):
    """
    The user’s RPM is for a FULL cycle (Left->Right->Left).
    e.g. 60 RPM => 1.0 second per full cycle => half-cycle is 0.5s.
    We'll do ± 'tolerance' around that half-cycle time.
    """
    full_cycle_sec = 60.0 / rpm
    half_cycle_sec = full_cycle_sec / 2.0
    min_time = half_cycle_sec*(1 - tolerance)
    max_time = half_cycle_sec*(1 + tolerance)
    logging.info(f"RPM={rpm}, half-cycle nominal={half_cycle_sec:.3f}s => range=({min_time:.3f}, {max_time:.3f})")
    return (min_time, max_time)

# -----------------------------------------------------------------------------
# LOADING BDF
# -----------------------------------------------------------------------------
def load_bdf_signals(file_path):
    raw_mne= mne.io.read_raw_bdf(file_path, preload=True)
    fs= raw_mne.info['sfreq']
    logging.info(f"Loaded BDF: {file_path} @ {fs:.1f} Hz")

    data_dict= {}
    for ax in ALL_AXES:
        if ax in raw_mne.info['ch_names']:
            data_dict[ax]= raw_mne.copy().pick(ax).get_data().flatten()
    return data_dict, fs

# -----------------------------------------------------------------------------
# DSP UTILS
# -----------------------------------------------------------------------------
def wavelet_denoise(signal, wavelet='db4', level=1):
    coeffs= pywt.wavedec(signal, wavelet, level=level)
    sigma_est= np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh= sigma_est * np.sqrt(2*np.log(len(signal)))
    coeffs_thresh= [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)

def low_pass_filter(signal, cutoff, fs, order=4):
    from scipy.signal import butter, filtfilt
    nyquist= 0.5* fs
    norm_cut= cutoff/ nyquist
    b,a= butter(order, norm_cut, btype='low', analog=False)
    return filtfilt(b,a,signal)

def apply_kalman_filter(signal):
    kf= KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=signal[0] if len(signal)>0 else 0,
        n_dim_obs=1
    )
    means,_= kf.filter(signal)
    return means.flatten()

def advanced_preprocessing(signal, fs,
                          use_wavelet=False,
                          wavelet_level=1,
                          polynomial_detrend=True,
                          cutoff_frequency=3):
    from scipy.signal import detrend as sp_detrend
    if use_wavelet and len(signal)>0:
        signal= wavelet_denoise(signal, level=wavelet_level)
    if polynomial_detrend and len(signal)>0:
        signal= sp_detrend(signal, type='linear')
    if len(signal)>0:
        signal= low_pass_filter(signal, cutoff_frequency, fs=fs)
    if len(signal)>0:
        signal= apply_kalman_filter(signal)
    return signal

def vector_magnitude(x,y,z):
    return np.sqrt(x**2 + y**2 + z**2)

# -----------------------------------------------------------------------------
# PEAK DETECTION & FEATURES
# -----------------------------------------------------------------------------
def adaptive_peak_detection(signal, fs,
                            initial_threshold,
                            prominence_scale_factor):
    from scipy.signal import find_peaks
    dist= int(0.7*fs)
    rolling_std= pd.Series(signal).rolling(window=500, center=True, min_periods=1).std()
    rolling_std= rolling_std.fillna(method='bfill').fillna(method='ffill').values
    local_arr= initial_threshold + (prominence_scale_factor* rolling_std)
    global_thr= np.median(local_arr)
    peaks,_= find_peaks(signal, prominence=global_thr, distance=dist)
    logging.debug(f"Adaptive peaks found: {len(peaks)} with threshold={global_thr:.5f}, dist={dist}")
    return np.array(peaks,dtype=int)

def extract_features(signal_segments, time_segment):
    feats= {}
    for axis, seg in signal_segments.items():
        feats[f'mean_{axis}']= np.mean(seg)
        feats[f'std_{axis}']= np.std(seg)
        feats[f'max_{axis}']= np.max(seg)
        feats[f'min_{axis}']= np.min(seg)
        feats[f'range_{axis}']= feats[f'max_{axis}'] - feats[f'min_{axis}']

    def axis_diff(a1,a2):
        if a1 in signal_segments and a2 in signal_segments:
            return np.mean(signal_segments[a1] - signal_segments[a2])
        return 0.0

    if 'GyroX' in signal_segments and 'GyroY' in signal_segments:
        feats['gyro_xy_diff']= axis_diff('GyroX','GyroY')
    if 'GyroX' in signal_segments and 'GyroZ' in signal_segments:
        feats['gyro_xz_diff']= axis_diff('GyroX','GyroZ')
    if 'GyroY' in signal_segments and 'GyroZ' in signal_segments:
        feats['gyro_yz_diff']= axis_diff('GyroY','GyroZ')

    if 'AccelerX' in signal_segments and 'AccelerY' in signal_segments:
        feats['accel_xy_diff']= axis_diff('AccelerX','AccelerY')
    if 'AccelerX' in signal_segments and 'AccelerZ' in signal_segments:
        feats['accel_xz_diff']= axis_diff('AccelerX','AccelerZ')
    if 'AccelerY' in signal_segments and 'AccelerZ' in signal_segments:
        feats['accel_yz_diff']= axis_diff('AccelerY','AccelerZ')

    if len(time_segment)>1:
        feats['duration']= time_segment[-1]- time_segment[0]
    else:
        feats['duration']= 0.0
    return feats

def enforce_alternating_movements(events):
    corrected= []
    last_mv= None
    for evt in events:
        if last_mv is None:
            corrected.append(evt)
            last_mv= evt['movement']
        else:
            if last_mv.startswith('Left') and evt['movement'].startswith('Left'):
                evt['movement']= 'Right Head Movement'
            elif last_mv.startswith('Right') and evt['movement'].startswith('Right'):
                evt['movement']= 'Left Head Movement'
            corrected.append(evt)
            last_mv= evt['movement']
    return corrected

# -----------------------------------------------------------------------------
# DETECT CYCLES (user-based RPM range)
# -----------------------------------------------------------------------------
def detect_cycles_rpm(events, rpm_range):
    """
    Accept only L->R pairs if duration is in [rpm_range[0], rpm_range[1]].
    """
    cycles= []
    invalid= []
    i=0
    while i< len(events)-1:
        e1= events[i]
        e2= events[i+1]
        if e1['movement'].startswith('Left') and e2['movement'].startswith('Right'):
            dur= e2['time']- e1['time']
            if rpm_range[0] <= dur <= rpm_range[1]:
                cycles.append({
                    'start_time': e1['time'],
                    'end_time': e2['time'],
                    'duration': dur,
                    'left_event': e1,
                    'right_event': e2
                })
            else:
                invalid.append({
                    'start_time': e1['time'],
                    'end_time': e2['time'],
                    'duration': dur
                })
            i+=1
        else:
            invalid.append(e1)
            i+=1
    if i== len(events)-1:
        invalid.append(events[-1])
    return cycles, invalid

def calculate_validity_score(cycles, invalid):
    total= len(cycles)+ len(invalid)
    return len(cycles)/ total if total>0 else 0.0

# -----------------------------------------------------------------------------
# XGBOOST UTILS
# -----------------------------------------------------------------------------
def train_xgboost_classifier_with_columns(X, y):
    columns= X.columns.tolist()
    le= LabelEncoder()
    y_enc= le.fit_transform(y)
    sc= StandardScaler()
    X_sc= sc.fit_transform(X)
    model= xgb.XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_sc, y_enc)
    return model, le, sc, columns

def xgboost_predict_proba(model, scaler, label_encoder, X, model_columns):
    X_aligned= X.reindex(columns=model_columns, fill_value=0.0)
    if len(X_aligned)==0:
        return [], []
    X_sc= scaler.transform(X_aligned)
    probs= model.predict_proba(X_sc)
    idx= np.argmax(probs, axis=1)
    preds= label_encoder.inverse_transform(idx)
    confs= np.max(probs, axis=1)
    return preds, confs

import xgboost
def dummy_xgb():
    model= xgboost.XGBClassifier()
    model.fit(np.zeros((2,2)), [0,1])
    le= LabelEncoder(); le.fit(["left","right"])
    sc= StandardScaler(); sc.fit(np.zeros((2,2)))
    columns= ["col1","col2"]
    return (model, le, sc, columns)

# -----------------------------------------------------------------------------
# SNIPPET-BASED XGBOOST
# -----------------------------------------------------------------------------
def build_balanced_xgb_model(raw_data, fs,
                             snippet_sec=10.0,
                             max_attempts=5):
    for attempt in range(max_attempts):
        model, le, sc, columns, got_both= build_xgb_once(raw_data, fs, snippet_sec)
        if got_both:
            return model, le, sc, columns
        logging.debug(f"Attempt {attempt+1}/{max_attempts}, didn't find both => retry.")
    logging.warning("Could not find both 'left' & 'right' => dummy xgb.")
    return dummy_xgb()

def build_xgb_once(raw_data, fs, snippet_sec):
    (model, le, sc, columns)= dummy_xgb()
    got_both= False

    if not raw_data:
        return model, le, sc, columns, got_both

    ch_name= next(iter(raw_data.keys()))
    total_len= len(raw_data[ch_name])
    seg_len= int(snippet_sec*fs)
    if seg_len>= total_len:
        st=0; ed= total_len
    else:
        max_st= total_len- seg_len
        random.seed(RANDOM_SEED)
        st= random.randint(0,max_st)
        ed= st+ seg_len

    snippet= {}
    for ch in raw_data:
        seg= raw_data[ch][st:ed]
        imp= SimpleImputer(strategy='mean')
        seg_imp= imp.fit_transform(seg.reshape(-1,1)).ravel()
        proc= low_pass_filter(seg_imp, 5, fs)
        proc= apply_kalman_filter(proc)
        snippet[ch]= proc

    detect_key= pick_detection_axis(snippet)
    if not detect_key:
        return model, le, sc, columns, got_both

    slen= len(snippet[detect_key])
    if slen<2:
        return model, le, sc, columns, got_both

    t_local= np.arange(slen)/fs
    pks= adaptive_peak_detection(snippet[detect_key], fs=fs,
                                 initial_threshold=0.005,
                                 prominence_scale_factor=1.5)
    if len(pks)==0:
        return model, le, sc, columns, got_both

    wsize= int(0.5*fs)
    feats_list= []
    labs= []
    for p in pks:
        s0= max(0,p-wsize)
        s1= min(slen,p+wsize)
        subd= {ax: snippet[ax][s0:s1] for ax in snippet}
        feats= extract_features(subd, t_local[s0:s1])
        feats_list.append(feats)

        mgx= feats.get('mean_GyroX',0)
        mgy= feats.get('mean_GyroY',0)
        if mgx>0 and mgy>0:
            labs.append('left')
        elif mgx<0 and mgy<0:
            labs.append('right')
        else:
            if abs(mgx)> abs(mgy):
                labs.append('left' if mgx>0 else 'right')
            else:
                labs.append('left' if mgy>0 else 'right')

    if not feats_list:
        return model, le, sc, columns, got_both

    df_feat= pd.DataFrame(feats_list)
    if len(df_feat)>1:
        iso= IsolationForest(random_state=RANDOM_SEED)
        outlbl= iso.fit_predict(df_feat)
        inl= np.where(outlbl==1)[0]
        df_feat= df_feat.iloc[inl].reset_index(drop=True)
        labs= [labs[i] for i in inl]

    if len(df_feat)<2:
        return model, le, sc, columns, got_both

    # check classes
    if len(set(labs))<2:
        return model, le, sc, columns, got_both

    m, le2, sc2, cols2= train_xgboost_classifier_with_columns(df_feat, labs)
    model, le, sc, columns= m, le2, sc2, cols2
    got_both= True

    # optional debug stats
    X_train, X_test, y_train, y_test= train_test_split(df_feat, labs, test_size=0.3, random_state=RANDOM_SEED)
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    sc_fit= StandardScaler()
    X_train_sc= sc_fit.fit_transform(X_train)
    X_test_sc= sc_fit.transform(X_test)
    le_enc= LabelEncoder()
    y_train_enc= le_enc.fit_transform(y_train)
    y_test_enc= le_enc.transform(y_test)
    model.fit(X_train_sc, y_train_enc)
    y_pred= model.predict(X_test_sc)
    acc= accuracy_score(y_test_enc,y_pred)
    logging.info(f"Snippet XGB attempt => accuracy= {acc:.2f}")
    if len(np.unique(y_test_enc))>1:
        from sklearn.metrics import classification_report
        logging.info("Snippet classification:\n"+
                     classification_report(y_test_enc,y_pred,target_names=le_enc.classes_))
    else:
        logging.info("Skipping classification_report (only one class in snippet test).")

    return model, le, sc, columns, got_both

def pick_detection_axis(seg_data):
    if 'CompoundSignal' in seg_data:
        return 'CompoundSignal'
    elif 'GyroMag' in seg_data:
        return 'GyroMag'
    elif 'AccelMag' in seg_data:
        return 'AccelMag'
    else:
        for ax in ['GyroX','AccelerX','GyroY','AccelerY','GyroZ','AccelerZ']:
            if ax in seg_data:
                return ax
    return None

# -----------------------------------------------------------------------------
def snippet_detect_events(raw_data, fs, duration_sec=10.0):
    """
    Minimal function for snippet-based detection => we log an auto range from it.
    Not used in final pass, just informational.
    """
    if not raw_data:
        return []
    ch_name= next(iter(raw_data.keys()))
    total_len= len(raw_data[ch_name])
    seg_len= int(duration_sec*fs)
    if seg_len>= total_len:
        st=0; ed= total_len
    else:
        max_st= total_len- seg_len
        random.seed(42)
        st= random.randint(0,max_st)
        ed= st+ seg_len

    snippet= {}
    for ax in raw_data:
        seg= raw_data[ax][st:ed]
        imp= SimpleImputer(strategy='mean')
        seg_imp= imp.fit_transform(seg.reshape(-1,1)).ravel()
        proc= low_pass_filter(seg_imp, cutoff=5, fs=fs)
        proc= apply_kalman_filter(proc)
        snippet[ax]= proc

    detect_key= pick_detection_axis(snippet)
    if not detect_key:
        return []
    slen= len(snippet[detect_key])
    if slen<2:
        return []

    t_local= np.arange(slen)/fs
    pks= adaptive_peak_detection(snippet[detect_key], fs=fs,
                                 initial_threshold=0.005,
                                 prominence_scale_factor=1.5)
    if len(pks)==0:
        return []
    wsize= int(0.5*fs)
    feats_list= []
    peak_times= []
    for p in pks:
        s0= max(0,p-wsize)
        s1= min(slen,p+wsize)
        subd= {ax: snippet[ax][s0:s1] for ax in snippet}
        feats= extract_features(subd, t_local[s0:s1])
        feats_list.append(feats)
        peak_times.append((st/fs)+ t_local[p])

    if not feats_list:
        return []
    df_feat= pd.DataFrame(feats_list)
    if len(df_feat)>1:
        iso= IsolationForest(random_state=RANDOM_SEED)
        outlbl= iso.fit_predict(df_feat)
        inl= np.where(outlbl==1)[0]
        df_feat= df_feat.iloc[inl].reset_index(drop=True)
        peak_times= np.array(peak_times)[inl]

    if len(df_feat)==0:
        return []

    # heuristic label
    labs= []
    for i in range(len(df_feat)):
        mgx= df_feat.get('mean_GyroX',0)[i]
        mgy= df_feat.get('mean_GyroY',0)[i]
        if mgx>0 and mgy>0:
            labs.append('left')
        elif mgx<0 and mgy<0:
            labs.append('right')
        else:
            if abs(mgx)> abs(mgy):
                labs.append('left' if mgx>0 else 'right')
            else:
                labs.append('left' if mgy>0 else 'right')

    events= []
    for i, lb in enumerate(labs):
        events.append({
            'time': peak_times[i],
            'movement': lb.capitalize()+" Head Movement"
        })
    cor= enforce_alternating_movements(events)
    return cor

def determine_interval_range(events):
    """
    For snippet-based auto range logging, not enforced in final pass.
    """
    intervals= []
    for i in range(len(events)-1):
        e1= events[i]
        e2= events[i+1]
        if e1['movement'].startswith('Left') and e2['movement'].startswith('Right'):
            dur= e2['time']- e1['time']
            intervals.append(dur)
    if not intervals:
        return (0.5,1.5)
    median_val= np.median(intervals)
    low= median_val*(1-RPM_TOLERANCE)
    high= median_val*(1+RPM_TOLERANCE)
    logging.info(f"Auto-detected median L->R interval= {median_val:.3f}s => snippet range=({low:.3f}, {high:.3f})")
    return (low, high)

def create_compound_signal_for_combo(data_copy, combo_name='CompoundSignal', axes=()):
    """
    Create a vector magnitude channel from the selected 'axes' in 'data_copy'
    and store it under combo_name (e.g., 'CompoundSignal').
    """
    if not axes:
        return
    missing= [ax for ax in axes if ax not in data_copy]
    if missing:
        logging.warning(f"Combo {axes} => missing axes {missing}, skipping compound.")
        return
    squares= []
    for ax in axes:
        squares.append(data_copy[ax]**2)
    sumsq= np.sum(squares, axis=0)
    magnitude= np.sqrt(sumsq)
    data_copy[combo_name]= magnitude

def save_event_times(cycles, prefix):
    if not cycles:
        logging.warning("No valid cycles => skipping save.")
        return
    left_times= [c['left_event']['time'] for c in cycles]
    right_times= [c['right_event']['time'] for c in cycles]

    outdir= "ProcessedGyroData"
    os.makedirs(outdir, exist_ok=True)
    left_file= os.path.join(outdir, f"{prefix}_left.txt")
    right_file= os.path.join(outdir, f"{prefix}_right.txt")
    np.savetxt(left_file, left_times, fmt='%.4f')
    np.savetxt(right_file, right_times, fmt='%.4f')
    logging.info(f"Saved left times => {left_file}, right => {right_file}")


# -----------------------------------------------------------------------------
# HELPER: SPLIT INTO LARGE CHUNKS
# -----------------------------------------------------------------------------
def split_into_large_chunks(signal_length, n_chunks=CHUNK_COUNT):
    """
    Splits the entire signal range [0, signal_length) into n_chunks segments.
    E.g. if length=450000 and n_chunks=3 => chunk size=150000 each.
    """
    chunk_size= signal_length//n_chunks
    segs= []
    for i in range(n_chunks):
        st= i*chunk_size
        ed= (i+1)*chunk_size if i<(n_chunks-1) else signal_length
        segs.append((st, ed))
    return segs

def sample_subchunks(chunk_start, chunk_end, fs, chunk_duration, n_subchunks):
    """
    For the chunk [chunk_start, chunk_end), randomly pick n_subchunks segments
    of length 'chunk_duration' in seconds, so we can run short heuristic detection
    for the Bayesian objective.
    """
    seg_len= chunk_end- chunk_start
    sub_size= int(chunk_duration*fs)
    max_start= seg_len- sub_size-1
    if max_start<1:
        return [(chunk_start, chunk_end)]
    random.seed(RANDOM_SEED)
    subs= []
    for _ in range(n_subchunks):
        st_local= random.randint(0, max_start)
        ed_local= st_local+ sub_size
        subs.append((chunk_start+st_local, chunk_start+ ed_local))
    return subs

# -----------------------------------------------------------------------------
def bayes_objective(params, raw_data, fs, subchunks):
    """
    Bayesian objective => we do a short detection to see how many L->R pairs
    appear (heuristically, ignoring the final RPM-based range).
    We maximize (# of cycles).
    """
    (init_thr, prom_scale, cutoff_freq, use_wav)= params
    total_valid= 0
    for (st,ed) in subchunks:
        evs, cyc= detect_peaks_and_cycles_heuristic(raw_data, fs, st, ed,
                                                    init_thr, prom_scale,
                                                    cutoff_freq, use_wav)
        total_valid+= len(cyc)
    return -total_valid

def detect_peaks_and_cycles_heuristic(raw_data, fs,
                                      start_idx, end_idx,
                                      init_thr, prom_scale,
                                      cutoff_freq, use_wav):
    seg_data= {}
    for ch in raw_data:
        seg= raw_data[ch][start_idx:end_idx]
        imp= SimpleImputer(strategy='mean')
        seg_imp= imp.fit_transform(seg.reshape(-1,1)).ravel()
        proc= advanced_preprocessing(seg_imp, fs=fs,
                                     use_wavelet=use_wav,
                                     cutoff_frequency=cutoff_freq)
        seg_data[ch]= proc

    detect_key= pick_detection_axis(seg_data)
    if not detect_key:
        return [],[]

    slen= len(seg_data[detect_key])
    if slen<2:
        return [],[]
    local_t= np.arange(slen)/fs

    pks= adaptive_peak_detection(seg_data[detect_key], fs=fs,
                                 initial_threshold=init_thr,
                                 prominence_scale_factor=prom_scale)
    if len(pks)==0:
        return [],[]

    wsize= int(0.5*fs)
    feats_list= []
    peak_times= []
    for p in pks:
        s0= max(0,p-wsize)
        s1= min(slen,p+wsize)
        subd= {ax: seg_data[ax][s0:s1] for ax in seg_data}
        feats= extract_features(subd, local_t[s0:s1])
        feats_list.append(feats)
        peak_times.append((start_idx/fs)+ local_t[p])

    if not feats_list:
        return [],[]
    df_feat= pd.DataFrame(feats_list)
    if len(df_feat)>1:
        iso= IsolationForest(random_state=RANDOM_SEED)
        outlbl= iso.fit_predict(df_feat)
        inl= np.where(outlbl==1)[0]
        df_feat= df_feat.iloc[inl].reset_index(drop=True)
        peak_times= np.array(peak_times)[inl]

    if len(df_feat)==0:
        return [],[]

    # heuristic labeling
    labs= []
    for i in range(len(df_feat)):
        mgx= df_feat.get('mean_GyroX',0)[i]
        mgy= df_feat.get('mean_GyroY',0)[i]
        if mgx>0 and mgy>0:
            labs.append('left')
        elif mgx<0 and mgy<0:
            labs.append('right')
        else:
            if abs(mgx)> abs(mgy):
                labs.append('left' if mgx>0 else 'right')
            else:
                labs.append('left' if mgy>0 else 'right')

    events= []
    for i, lb in enumerate(labs):
        events.append({
            'time': peak_times[i],
            'movement': lb.capitalize()+" Head Movement"
        })
    cor= enforce_alternating_movements(events)

    # define cycles as all L->R pairs (no range check here)
    cyc= []
    i=0
    while i< len(cor)-1:
        e1= cor[i]
        e2= cor[i+1]
        if e1['movement'].startswith('Left') and e2['movement'].startswith('Right'):
            cyc.append({'start_time': e1['time'], 'end_time': e2['time']})
            i+=1
        i+=1
    return cor, cyc

# -----------------------------------------------------------------------------
# FINAL PASS
# -----------------------------------------------------------------------------
def run_detection_on_chunk(raw_data, fs, chunk_indices, params,
                           xgb_model, xgb_le, xgb_scaler, model_columns,
                           rpm_range):
    from sklearn.impute import SimpleImputer
    (start_idx, end_idx)= chunk_indices
    seg_data= {}
    for ch in raw_data:
        seg= raw_data[ch][start_idx:end_idx]
        imp= SimpleImputer(strategy='mean')
        seg_imp= imp.fit_transform(seg.reshape(-1,1)).ravel()
        proc= advanced_preprocessing(seg_imp, fs=fs,
                                     use_wavelet=params['use_wavelet'],
                                     cutoff_frequency=params['cutoff_frequency'])
        seg_data[ch]= proc

    detect_key= pick_detection_axis(seg_data)
    if not detect_key:
        logging.debug("No valid detection axis => 0 events.")
        return [],[],[]

    slen= len(seg_data[detect_key])
    local_t= np.arange(slen)/fs
    pks= adaptive_peak_detection(seg_data[detect_key], fs=fs,
                                 initial_threshold=params['initial_threshold'],
                                 prominence_scale_factor=params['prominence_scale_factor'])
    logging.debug(f"Chunk detection => found {len(pks)} peaks.")
    if len(pks)==0:
        return [],[],[]

    wsize= int(0.5*fs)
    feats_list= []
    peak_times= []
    for p in pks:
        s0= max(0,p-wsize)
        s1= min(slen,p+wsize)
        subd= {ax: seg_data[ax][s0:s1] for ax in seg_data}
        feats= extract_features(subd, local_t[s0:s1])
        feats_list.append(feats)
        peak_times.append((start_idx/fs)+ local_t[p])

    if not feats_list:
        logging.debug("No features => 0 events.")
        return [],[],[]

    df_feat= pd.DataFrame(feats_list)
    if len(df_feat)>1:
        iso= IsolationForest(random_state=RANDOM_SEED)
        outlbl= iso.fit_predict(df_feat)
        inl= np.where(outlbl==1)[0]
        df_feat= df_feat.iloc[inl].reset_index(drop=True)
        peak_times= np.array(peak_times)[inl]

    if len(df_feat)==0:
        logging.debug("All peaks outliers => 0 events.")
        return [],[],[]

    pred_lbl, confs= xgboost_predict_proba(xgb_model, xgb_scaler, xgb_le, df_feat, model_columns)
    accepted_idx= np.where(confs>= CONFIDENCE_THRESHOLD)[0]
    logging.debug(f"XGBoost => total events={len(pred_lbl)}, accepted={len(accepted_idx)} (conf>= {CONFIDENCE_THRESHOLD})")

    events= []
    for i in range(len(pred_lbl)):
        logging.debug(f" event{i} => label={pred_lbl[i]}, conf={confs[i]:.3f}")
    for i in accepted_idx:
        lb= pred_lbl[i]
        events.append({
            'time': peak_times[i],
            'movement': lb.capitalize()+" Head Movement"
        })

    cor= enforce_alternating_movements(events)
    cyc, inv= detect_cycles_rpm(cor, rpm_range)
    logging.debug(f"Final chunk => events={len(events)}, cycles={len(cyc)}, invalid={len(inv)}")
    return events, cyc, inv

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    try:
        # 1) Prompt for BDF
        fname= get_filename()
        full_path= os.path.abspath(fname)

        # 2) Prompt for user RPM
        rpm_val= get_rpm()
        rpm_range= compute_rpm_range(rpm_val, tolerance=RPM_TOLERANCE)

        # 3) Load data
        raw_data, fs= load_bdf_signals(full_path)
        if not raw_data:
            logging.error("No valid gyro/accel channels => abort.")
            return

        # snippet detection => log auto-range
        snippet_evts= snippet_detect_events(raw_data, fs, SNIPPET_SECONDS)
        if snippet_evts:
            determine_interval_range(snippet_evts)
        else:
            logging.debug("snippet_detect_events => no events found.")

        # build combos
        combo_list= []
        for r in range(1, MAX_AXES_COMBO_SIZE+1):
            for combo in itertools.combinations(ALL_AXES, r):
                if all(ax in raw_data for ax in combo):
                    combo_list.append(combo)
        logging.info(f"Axis combos to test: {combo_list}")

        best_combo= None
        best_cycles_count= 0
        best_cycles= []
        best_invalid= []
        best_data_for_plot= None

        from copy import deepcopy

        # Evaluate combos
        for combo in combo_list:
            logging.info(f"\n===== Testing combination: {combo} =====")
            data_copy= deepcopy(raw_data)

            # create "CompoundSignal" from these axes
            create_compound_signal_for_combo(data_copy, 'CompoundSignal', combo)

            # snippet-based XGBoost
            xgb_model, xgb_le, xgb_scaler, model_cols= build_balanced_xgb_model(
                data_copy, fs, snippet_sec=SNIPPET_SECONDS, max_attempts=MAX_SNIPPET_ATTEMPTS
            )

            ch_name= next(iter(data_copy.keys()))
            total_len= len(data_copy[ch_name])
            big_chunks= split_into_large_chunks(total_len, CHUNK_COUNT)

            combo_cycles= []
            combo_invalid= []
            combo_events= []

            # chunk-based approach
            for i,(cstart,cend) in enumerate(big_chunks):
                logging.info(f" [Combo {combo}, chunk {i+1}/{CHUNK_COUNT} => {cstart}:{cend}]")
                subs= sample_subchunks(cstart, cend, fs, SUBCHUNK_DURATION, SUBCHUNK_COUNT)

                def objective_fn(params_list):
                    return bayes_objective(params_list, data_copy, fs, subs)

                res= gp_minimize(
                    func=objective_fn,
                    dimensions=SEARCH_SPACE,
                    n_calls=BAYES_CALLS,
                    n_initial_points=BAYES_CALLS,
                    random_state=RANDOM_SEED
                )
                best_params= {
                    'initial_threshold': res.x[0],
                    'prominence_scale_factor': res.x[1],
                    'cutoff_frequency': res.x[2],
                    'use_wavelet': res.x[3]
                }
                best_score= -res.fun
                logging.info(f"  => best params for chunk {i+1}: {best_params}, valid cyc={best_score}")

                # final pass => only keep durations in rpm_range
                c_evts, c_cyc, c_inv= run_detection_on_chunk(
                    data_copy, fs, (cstart,cend),
                    best_params,
                    xgb_model, xgb_le, xgb_scaler, model_cols,
                    rpm_range
                )
                combo_events.extend(c_evts)
                combo_cycles.extend(c_cyc)
                combo_invalid.extend(c_inv)

            cyc_count= len(combo_cycles)
            logging.info(f" [COMBO={combo}] => final cycles= {cyc_count}")
            if cyc_count> best_cycles_count:
                best_cycles_count= cyc_count
                best_combo= combo
                best_cycles= combo_cycles
                best_invalid= combo_invalid
                best_data_for_plot= data_copy

        logging.info(f"\nBEST COMBO= {best_combo}, yield {best_cycles_count} cycles.")

        if not best_data_for_plot:
            logging.warning("No combos produced cycles => no plot.")
            return

        # Plot best
        if 'CompoundSignal' in best_data_for_plot:
            detect_axis= 'CompoundSignal'
        else:
            if best_combo:
                detect_axis= best_combo[0]
            else:
                detect_axis= next(iter(best_data_for_plot.keys()))

        t_global= np.arange(len(best_data_for_plot[detect_axis]))/fs

        plt.figure(figsize=(12,6))
        plt.plot(t_global, best_data_for_plot[detect_axis], label=f"{detect_axis} for best combo={best_combo}")
        for c in best_cycles:
            plt.axvspan(c['start_time'], c['end_time'], color='green', alpha=0.3)
        for inv in best_invalid:
            if 'start_time' in inv:
                plt.axvspan(inv['start_time'], inv['end_time'], color='red', alpha=0.3)

        plt.title(f"BEST COMBO={best_combo}, Found {best_cycles_count} cycles\n(RPM half-cycle range= {rpm_range} )")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Amplitude")
        plt.legend()
        plt.show()

        # final stats
        if best_cycles:
            durations= [c['duration'] for c in best_cycles]
            avg_dur= np.mean(durations)
            std_dur= np.std(durations)
            logging.info(f"STAT: Found {len(durations)} final cycles in best combo.")
            logging.info(f"STAT: Mean cycle duration= {avg_dur:.3f}s, std= {std_dur:.3f}s")

        base_file= os.path.splitext(os.path.basename(fname))[0]
        save_event_times(best_cycles, base_file)
        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"Error: {e}")


if __name__=="__main__":
    main()
