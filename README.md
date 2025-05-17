BDF Signal Processing and Head Movement Detection

Developed by Dr. Marcelo Bigliassi (Florida International University)


This Python application performs advanced signal processing and machine learning classification on BDF files (BioSemi Data Format). It is specifically designed to detect alternating left and right head movements using gyroscope and accelerometer data.

Features

Loads .bdf files using MNE
Performs adaptive preprocessing (wavelet denoising, detrending, low-pass filtering, Kalman filtering)
Automatically builds a compound signal from selected axis combinations
Uses Bayesian optimization to tune peak detection parameters
Employs XGBoost for classification of detected movements
Enforces alternating left-right movement logic
Detects valid movement cycles within user-defined RPM constraints
Supports multiprocessing for performance scaling
Visualizes the best-performing detection combination with cycle spans
Outputs timestamps of detected left and right movements into text files
Dependencies

This project requires Python 3.8+ and the following libraries:

mne
numpy
pandas
scikit-learn
scipy
matplotlib
xgboost
pykalman
pywt
scikit-optimize
You can install all dependencies using:

pip install -r requirements.txt
Or manually:

pip install mne numpy pandas scikit-learn scipy matplotlib xgboost pykalman pywt scikit-optimize
Usage

Run the script using:

python your_script_name.py
You will be prompted to:

Enter the path to a .bdf file.
Specify the nominal RPM (cycles per minute) for your movement task.
The script will:

Load and preprocess the data.
Evaluate combinations of axes (gyro and accel) for signal quality.
Run a hybrid heuristic/ML pipeline to detect head movements.
Train an XGBoost classifier using labeled signal snippets.
Detect valid left-right movement cycles across the full signal.
Visualize results and export timestamps to text files in the ProcessedGyroData folder.
Output

Visualization of best-performing signal axis or compound
ProcessedGyroData/<filename>_left.txt and <filename>_right.txt containing timestamps of detected movements
Console logs detailing accuracy, confusion matrix, and validity metrics
Customization

The following configuration options can be modified in the script:

CHUNK_COUNT: Number of large time segments for analysis
SUBCHUNK_DURATION: Duration (in seconds) of sub-snippets used for Bayesian optimization
RPM_TOLERANCE: Acceptable deviation in expected cycle duration
CONFIDENCE_THRESHOLD: Minimum probability for XGBoost classification acceptance
MAX_AXES_COMBO_SIZE: Maximum number of axes to combine when creating compound signals
Notes

Ensure that your .bdf file includes correctly labeled gyroscope and accelerometer channels.
The script assumes alternating left/right head movements and enforces this structure for cycle detection.
It is optimized for large datasets and can skip low-performing signal segments.

License

This project is for research and educational use only.
