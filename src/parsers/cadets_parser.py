from pathlib import Path
import json
from typing import Dict, List, Optional, Any, Generator
import pandas as pd
from datetime import datetime
import yaml
from dataclasses import dataclass
from tqdm import tqdm

from src.utils.logging_utils import setup_logger

@dataclass
class EventMapping:
    """Mapping configurations for event types"""
    event_to_x: Dict[str, str] = None
    event_to_type: Dict[str, str] = None

    def __post_init__(self):
        self.event_to_x = {
            'EVENT_EXECUTE': 'execute', 'EVENT_FORK': 'fork',
            'EVENT_MODIFY_FILE_ATTRIBUTES': 'modify', 'EVENT_READ': 'read',
            'EVENT_RECVFROM': 'receive', 'EVENT_SENDTO': 'send',
            'EVENT_WRITE': 'write', 'EVENT_CLONE': 'fork', 'TcpIp/Recv': 'receive',
            'TcpIp/Send': 'send', 'FileIO/Read': 'read', 'FileIO/Write': 'write',
            'Process/Start': 'fork', 'Image/Load': 'execute'
        }
        
        self.event_to_type = {
            'execute': 'file', 'fork': 'process', 'modify': 'file',
            'read': 'file', 'receive': 'socket', 'send': 'socket',
            'write': 'file'
        }

class CadetsParser:
    """Parser for CADETS dataset"""
    
    def __init__(self, config_path: str):
        """Initialize parser with configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.event_mapping = EventMapping()
        self.logger = setup_logger(
            'cadets_parser',
            Path(self.config['paths']['logs_dir'])
        )
        self._counter = 0  # Used for generating sequential UUIDs
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _process_netflow(self, record: Dict) -> tuple[str, str]:
        """Process a NetFlow record
        
        Args:
            record: NetFlow record dictionary
            
        Returns:
            Tuple of (UUID, socket string)
        """
        socket = f"{record['remoteAddress']}:{record['remotePort']}"
        return record['uuid'], socket

    def _process_event(self, record: Dict) -> Optional[Dict]:
        """Process an event record
        
        Args:
            record: Event record dictionary
            
        Returns:
            Processed event dictionary or None if event should be skipped
        """
        if (record['type'] not in self.event_mapping.event_to_x or 
            record['subject'] is None or 
            record['predicateObject'] is None):
            return None

        return {
            'processUUID': record['subject']['com.bbn.tc.schema.avro.cdm18.UUID'],
            'objectUUID': record['predicateObject']['com.bbn.tc.schema.avro.cdm18.UUID'],
            'event': self.event_mapping.event_to_x[record['type']],
            'timestamp': record['timestampNanos'],
            'uuid': str(record['uuid']),
            'processName': record['properties']['map'].get('exec', 'NA').split('/')[-1],
            'objectData': (record.get('predicateObjectPath') or {}).get('string', 'NA'),
            'objectType': self.event_mapping.event_to_type[
                self.event_mapping.event_to_x[record['type']]
            ]
        }

    def parse_file(self, file_path: Path) -> tuple[Dict[str, str], List[Dict]]:
        """Parse a single CADETS log file
        
        Args:
            file_path: Path to log file
            
        Returns:
            Tuple of (network mapping dict, event logs list)
        """
        net_map = {}
        event_logs = []
        
        try:
            with open(file_path, encoding="utf8") as f:
                for line in tqdm(f, desc=f"Processing {file_path.name}"):
                    try:
                        data = json.loads(line)['datum']
                        
                        if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in data:
                            uuid, socket = self._process_netflow(
                                data['com.bbn.tc.schema.avro.cdm18.NetFlowObject']
                            )
                            net_map[uuid] = socket
                            
                        elif 'com.bbn.tc.schema.avro.cdm18.Event' in data:
                            event = self._process_event(
                                data['com.bbn.tc.schema.avro.cdm18.Event']
                            )
                            if event:
                                event_logs.append(event)
                                
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping malformed JSON line in {file_path}")
                    except KeyError as e:
                        self.logger.warning(f"Missing key {e} in record, skipping")
                        
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
            
        self.logger.info(f"Processed {len(event_logs)} events from {file_path}")
        return net_map, event_logs

    def parse_directory(self, source: str):
        """Parse all files in the raw data directory
        
        Args:
            source: Name of the dataset (e.g., 'cadets')
        """
        data_dir = Path(self.config['paths']['data_dir']) / source / 'raw_data'
        output_dir = Path(self.config['paths']['data_dir']) / source
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize output files
        with open(output_dir / 'parsed_events.json', 'w') as f:
            pass
            
        net_map = {}
        total_events = 0
        
        for file_path in data_dir.glob('*'):
            if file_path.is_file():
                self.logger.info(f"Processing {file_path}")
                file_net_map, events = self.parse_file(file_path)
                net_map.update(file_net_map)
                
                # Append events to output file
                with open(output_dir / 'parsed_events.json', 'a') as f:
                    for event in events:
                        f.write(json.dumps(event) + '\n')
                        
                total_events += len(events)
                
        # Save network mapping
        with open(output_dir / 'net_map.json', 'w') as f:
            json.dump(net_map, f)
            
        self.logger.info(f"Total events processed: {total_events}")
        self.logger.info(f"Network mappings created: {len(net_map)}") 

    def _process_mapped_event(self, log: Dict) -> Dict:
        """Process a single event during the mapping phase
        
        Args:
            log: Event log dictionary
            
        Returns:
            Processed event dictionary
        """
        log['uuid'] = str(self._counter)
        self._counter += 1
        
        if log['event'] == 'fork':
            log['objectData'] = log['processName']
        elif log['event'] in ['send', 'receive']:
            log['objectData'] = self.net_map.get(log['objectUUID'], 'NA')
        elif (log['event'] in ['read', 'write']) and (log['objectUUID'] in self.net_map):
            log['objectData'] = self.net_map[log['objectUUID']]
            log['objectType'] = 'socket'
            log['event'] = 'send' if log['event'] == 'write' else 'receive'
            
        return log

    def _save_batch(self, batch: List[Dict], output_file: Path, mode: str = 'a'):
        """Save a batch of events to file
        
        Args:
            batch: List of event dictionaries
            output_file: Path to output file
            mode: File open mode ('a' for append, 'w' for write)
        """
        with open(output_file, mode) as f:
            for log in batch:
                if log['objectData'] != 'NA' and log['objectData'] != 'NA:0':
                    f.write(json.dumps(log) + '\n')

    def map_events(self, source: str, timesplit: Optional[str] = None):
        """Map events and split into train/test sets
        
        Args:
            source: Name of the dataset
            timesplit: Timestamp for train/test split. If None, uses config default
        """
        data_dir = Path(self.config['paths']['data_dir']) / source
        batch_size = self.config['parsing']['batch_size']
        
        # Load network mapping
        try:
            with open(data_dir / 'net_map.json') as f:
                self.net_map = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Network mapping file not found. Run parse_directory first.")
            raise
            
        # Convert timesplit to nanoseconds
        if timesplit is None:
            timesplit = self.config['parsing']['default_timesplit']
        split_time = pd.to_datetime(timesplit).value
        
        # Initialize output files
        train_file = data_dir / 'train_logs.json'
        test_file = data_dir / 'test_logs.json'
        for file in [train_file, test_file]:
            with open(file, 'w') as f:
                pass
        
        # Process events
        train_batch, test_batch = [], []
        total_train, total_test = 0, 0
        self._counter = 0  # Reset counter for sequential UUIDs
        
        try:
            with open(data_dir / 'parsed_events.json') as f:
                for line in tqdm(f, desc="Mapping events"):
                    try:
                        log = json.loads(line)
                        processed_log = self._process_mapped_event(log)
                        
                        # Split based on timestamp
                        if log['timestamp'] < split_time:
                            train_batch.append(processed_log)
                            if len(train_batch) >= batch_size:
                                self._save_batch(train_batch, train_file)
                                total_train += len(train_batch)
                                train_batch = []
                        else:
                            test_batch.append(processed_log)
                            if len(test_batch) >= batch_size:
                                self._save_batch(test_batch, test_file)
                                total_test += len(test_batch)
                                test_batch = []
                                
                    except json.JSONDecodeError:
                        self.logger.warning("Skipping malformed JSON line")
                    except KeyError as e:
                        self.logger.warning(f"Missing key {e} in record, skipping")
            
            # Save remaining batches
            if train_batch:
                self._save_batch(train_batch, train_file)
                total_train += len(train_batch)
            if test_batch:
                self._save_batch(test_batch, test_file)
                total_test += len(test_batch)
                
        except Exception as e:
            self.logger.error(f"Error during event mapping: {str(e)}")
            raise
            
        self.logger.info(f"Total train events: {total_train}")
        self.logger.info(f"Total test events: {total_test}")
        
        # Convert to DataFrame and save as pickle
        self._save_to_pickle(train_file, test_file, source)

    def _save_to_pickle(self, train_file: Path, test_file: Path, source: str):
        """Convert JSON files to pickle format
        
        Args:
            train_file: Path to train JSON file
            test_file: Path to test JSON file
            source: Name of the dataset
        """
        data_dir = Path(self.config['paths']['data_dir']) / source
        
        # Process train set
        df = pd.read_json(train_file, lines=True)
        df.to_pickle(data_dir / 'train_logs.pkl')
        self.logger.info(f'Train: {df.shape[0]} logs saved to train_logs.pkl')
        del df
        
        # Process test set
        df = pd.read_json(test_file, lines=True)
        
        # Apply fixes for broken processes
        broken_processes = self.config['parsing']['broken_processes']
        mask = (df['event'] != 'fork') & (df['objectData'].isin(broken_processes))
        df.loc[mask, 'objectData'] = '/tmp/' + df['objectData']
        
        df.to_pickle(data_dir / 'test_logs.pkl')
        self.logger.info(f'Test: {df.shape[0]} logs saved to test_logs.pkl')

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process CADETS dataset')
    parser.add_argument('source', type=str, help='Dataset name (e.g., cadets)')
    parser.add_argument('--config', type=str, default='config/parser_config.yaml',
                      help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['parse', 'map', 'both'],
                      default='both', help='Operation mode')
    parser.add_argument('--timesplit', type=str,
                      help='Timestamp for train/test split (optional)')
    
    args = parser.parse_args()
    
    cadets_parser = CadetsParser(args.config)
    
    if args.mode in ('parse', 'both'):
        cadets_parser.parse_directory(args.source)
    
    if args.mode in ('map', 'both'):
        cadets_parser.map_events(args.source, args.timesplit)

if __name__ == '__main__':
    main() 