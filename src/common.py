import collections

PROJECTS_TO_EXCLUDE = []

RESULTS_DIR = '../results'
RAW_DDIR = '../data/raw'
PROCESSED_DDIR = '../data/processed'
LOAD_TO_EXCLUDE = 'features_to_exclude'
LOAD_CATEGORICAL = 'categorical_features'
LOAD_CATEGORICAL_OPTS = 'categorical_feature_values'
LOAD_CLASS = 'class'
CONFIGS = 'configurations'
TECHNIQUES = 'techniques'

SF_ALPHA = 0.05
RF_RANDOM_STATE = 42

NT_ModelResults = collections.namedtuple('NT_ModelResults', ['actuals', 'predictions', 'percentiles'])
