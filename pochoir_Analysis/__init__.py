'''
Post-run analysis helpers for pochoir stores.

This package holds tools that read a finished store directory and
compare it against a reference run.  Nothing here is imported by the
solver; it exists so validation of a physics change (see
pochoir_Analysis.compare_stores) does not have to live in ad-hoc
scratch scripts.
'''
