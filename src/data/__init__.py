"""
src/data/ — Space Sensing data registration and preprocessing stubs.

This directory holds Space Sensing-specific data utilities.  Dataset
registration (the ``data_dict`` mapping used by the training framework) lives
in:

    src/train_framework/data/__init__.py

which is the authoritative registry for the ``--dataset_use`` argument.

Scripts in this directory (e.g. preprocess_vsi590k.py in ../preprocess/) are
responsible for converting raw VSI-Bench / ScanNet data into the JSON/JSONL
annotation format expected by the GUIDE data_processor.
"""
