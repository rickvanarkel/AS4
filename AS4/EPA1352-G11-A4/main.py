"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
############################### This will run all Python files! ######################################
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

filenames = ['preparing_data.py', 'htm_cleaning.py', 'including_traffic_data.py', 'analysis_criticality.py', 'analysis_vulnerability.py']

for filename in filenames:
    with open(filename) as f:
        code = compile(f.read(), filename, 'exec')
        exec(code)

