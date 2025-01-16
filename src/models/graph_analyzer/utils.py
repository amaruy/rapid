import pandas as pd
import numpy as np

def find_ttps(row):
    """Find Tactics, Techniques, and Procedures (TTPs) based on event data."""
    per_process_ttps = {}
    
    if row['event'] in ['receive']:
        [ip,port] = str(row['objectData']).split(':')
        port = port.replace(",", "")
        if int(port) > 9999:
            if row['processUUID'] not in per_process_ttps:
                per_process_ttps[row['processUUID']] = {'T1595'}
                return 'T1595'
            else:
                if port in ['22', '80', '443']:
                    per_process_ttps[row['processUUID']].add('T1071')
                    return 'T1071'
                else:
                    per_process_ttps[row['processUUID']].add('T1571')
                    return 'T1571'
        else:
            return ''
    elif row['event'] in ['send']:
        [ip,port] = str(row['objectData']).split(':')
        port = port.replace(",", "")
        if int(port) <= 9999:
            if row['processUUID'] not in per_process_ttps:
                per_process_ttps[row['processUUID']] = {'T1189'}
                return 'T1189'
            else:
                if 'T1071' in per_process_ttps[row['processUUID']] or 'T1571' in per_process_ttps[row['processUUID']]:
                    per_process_ttps[row['processUUID']].add('T1041')
                    return 'T1041'
                else:
                    per_process_ttps[row['processUUID']].add('T1048')
                    return 'T1048'
        else:
            return ''
    elif row['event'] in ['write']:
        if row['processUUID'] in per_process_ttps:
            if 'T1189' in per_process_ttps[row['processUUID']] or 'T1595' in per_process_ttps[row['processUUID']]:
                per_process_ttps[row['processUUID']].add('T1105')
                return 'T1105'
            else:
                return ''
        else:
            return ''
    elif row['event'] in ['modify']:
        dirs = row['objectData'].split('/')
        if 'tmp' in dirs:
            if row['processUUID'] not in per_process_ttps:
                per_process_ttps[row['processUUID']] = {'T1222'}
                return 'T1222'
            else:
                per_process_ttps[row['processUUID']].add('T1222')
                return 'T1222'
        else:
            return ''
    elif row['event'] in ['execute']:
        dirs = row['objectData'].split('/')
        if 'tmp' in dirs:
            if row['processUUID'] not in per_process_ttps:
                per_process_ttps[row['processUUID']] = {'T1222'}
                return 'T1203'
            else:
                per_process_ttps[row['processUUID']].add('T1222')
                return 'T1203'
        else:
            return ''
    else:
        return ''

def filter_files(data: pd.DataFrame) -> set:
    """Filter important files based on dataflow and process counts."""
    file_data = data[data['objectType'] == 'file']
    grouped = file_data.groupby('objectData')
    dataflow_unique_counts = grouped['dataflow'].nunique()
    process_unique_counts = grouped['processUUID'].nunique()
    important_files = (dataflow_unique_counts > 1) | (process_unique_counts >= 2)
    return set(important_files[important_files].index)

def filter_processes(data: pd.DataFrame) -> set:
    """Filter important processes based on dataflow patterns."""
    multiple_dataflows = data.groupby('processUUID')['dataflow'].nunique() > 1
    contains_outward = data['dataflow'].eq('outward')
    in_objectUUID = data['processUUID'].isin(data['objectUUID'])
    outward_and_in_objectUUID = data[contains_outward & in_objectUUID].groupby('processUUID').size() > 0
    important_processes = multiple_dataflows | outward_and_in_objectUUID
    return set(important_processes[important_processes].index)

def df_to_edge_list(df: pd.DataFrame) -> list:
    """Convert DataFrame to edge list format."""
    df = df.copy()
    df['object'] = df.apply(lambda x: x['objectUUID'] if x['event'] == 'fork' else x['objectData'], axis=1)

    edges = []
    for subject, event, object in df[['processUUID', 'event', 'object']].itertuples(index=False):
        if event in ['send', 'write', 'modify', 'fork']:  # outward
            edges.append((subject, object))
        else:  # inward
            edges.append((object, subject))

    return edges 