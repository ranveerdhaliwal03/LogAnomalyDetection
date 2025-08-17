
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import re

class LogParser:
    """Parse HDFS logs using Drain3 to extract log templates and template IDs."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Drain3 template miner with configuration.
        
        Args:
            config: Configuration dictionary containing Drain3 settings
        """
        self.config = config
        self.persistence_file = config['drain3']['persistence_file']
        self.persistence = FilePersistence(self.persistence_file)
        
        # Configure Drain3 from config
        drain_config = config['drain3']
        config_obj = TemplateMinerConfig()
        config_obj.drain_sim_th = drain_config['similarity_threshold']
        config_obj.drain_depth = drain_config['tree_depth']
        config_obj.drain_max_children = drain_config['max_children']
        config_obj.drain_max_clusters = drain_config['max_clusters']
        
        self.template_miner = TemplateMiner(self.persistence, config_obj)
        self.template_count = 0
        
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single HDFS log line and extract structured information.
        
        Args:
            line: Raw log line from HDFS log file
            
        Returns:
            Dictionary with parsed log information or None if parsing fails
        """
        line = line.strip()
        if not line:
            return None
            
        # HDFS log format: MMDDYY HHMMSS milliseconds INFO component: message
        # Example: 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
        
        pattern = r'(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+): (.+)'
        match = re.match(pattern, line)
        
        if not match:
            return None
            
        date_str, time_str, milliseconds, level, component, message = match.groups()
        
        try:
            # Parse date: MMDDYY format (assuming 2008 as base year)
            month = int(date_str[:2])
            day = int(date_str[2:4])
            year = 2000 + int(date_str[4:6])
            
            # Parse time: HHMMSS format
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            timestamp = datetime(2008, month, day, hour, minute, second, int(milliseconds) * 1000)
        except Exception as e:
            print(f"Warning: Failed to parse timestamp from '{line}': {e}")
            timestamp = None
            
        # Extract block ID if present
        block_id = None
        block_match = re.search(r'blk_(-?\d+)', message)
        if block_match:
            block_id = block_match.group(1)
            
        # Use Drain3 to get template ID and template string
        drain_result = self.template_miner.add_log_message(line)
        template_id = drain_result['cluster_id']
        template_str = drain_result['template_mined']
        
        # Track unique templates
        if template_id > self.template_count:
            self.template_count = template_id
            
        return {
            'timestamp': timestamp,
            'level': level,
            'component': component,
            'message': message,
            'block_id': block_id,
            'raw_line': line,
            'template_id': template_id,
            'template_str': template_str
        }
    
    def parse_logs(self, log_file_path: str, max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse the entire log file using Drain3.
        
        Args:
            log_file_path: Path to HDFS log file
            max_lines: Maximum number of lines to parse (for testing)
            
        Returns:
            List of parsed log dictionaries
        """
        print(f"Parsing logs from: {log_file_path}")
        
        parsed_logs = []
        total_lines = 0
        parsed_count = 0
        
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                    
                total_lines += 1
                parsed_log = self.parse_log_line(line)
                if parsed_log:
                    parsed_logs.append(parsed_log)
                    parsed_count += 1
                    
                if total_lines % 10000 == 0:
                    print(f"Processed {total_lines:,} lines, parsed {parsed_count:,} successfully")
        
        print(f"\nParsing complete!")
        print(f"Total lines processed: {total_lines:,}")
        print(f"Successfully parsed: {parsed_count:,}")
        print(f"Unique templates found: {self.template_count}")
        
        return parsed_logs