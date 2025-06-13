"""
Alert management utilities for CIFAR-10 API.
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertManager:
    """Alert management for updating recipients and thresholds."""
    
    def __init__(self, config_dir: str = "monitoring"):
        self.config_dir = Path(config_dir)
        self.alertmanager_config_path = self.config_dir / "alertmanager" / "alertmanager.yml"
        self.alert_rules_path = self.config_dir / "prometheus" / "alert_rules.yml"
        
        # Default alert recipients
        self.default_recipients = [
            "22520630@gm.uit.edu.vn",
            "bgkhanh666@gmail.com"
        ]
    
    def update_alert_recipients(self, recipients: List[str], severity: str = "both"):
        """
        Update alert recipients list.
        
        Args:
            recipients: List of email addresses
            severity: "critical", "warning", or "both"
        """
        try:
            # Read current alertmanager config
            with open(self.alertmanager_config_path, 'r') as f:
                config_content = f.read()
            
            # For simplicity, we'll update the config by string replacement
            # In production, you might want to use a YAML parser
            
            if severity in ["critical", "both"]:
                # Update critical alerts recipients
                recipients_str = ", ".join([f"'{email}'" for email in recipients])
                config_content = self._update_config_recipients(
                    config_content, "critical-alerts", recipients_str
                )
            
            if severity in ["warning", "both"]:
                # Update warning alerts recipients
                recipients_str = ", ".join([f"'{email}'" for email in recipients])
                config_content = self._update_config_recipients(
                    config_content, "warning-alerts", recipients_str
                )
            
            # Write updated config
            with open(self.alertmanager_config_path, 'w') as f:
                f.write(config_content)
            
            logger.info(f"Updated {severity} alert recipients: {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update alert recipients: {e}")
            return False
    
    def _update_config_recipients(self, config_content: str, receiver_name: str, recipients: str):
        """Helper method to update recipients in config."""
        # This is a simplified approach - in production, use proper YAML parsing
        lines = config_content.split('\n')
        in_receiver = False
        
        for i, line in enumerate(lines):
            if f"name: '{receiver_name}'" in line:
                in_receiver = True
            elif in_receiver and 'to:' in line:
                # Update the recipients line
                indent = len(line) - len(line.lstrip())
                lines[i] = ' ' * indent + f"to: {recipients}"
                in_receiver = False
        
        return '\n'.join(lines)
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]):
        """
        Update alert thresholds.
        
        Args:
            thresholds: Dictionary of threshold values
                e.g., {"cpu_threshold": 80, "memory_threshold": 70}
        """
        try:
            # Read current alert rules
            with open(self.alert_rules_path, 'r') as f:
                rules_content = f.read()
            
            # Update thresholds
            threshold_mappings = {
                "cpu_threshold": ("system_cpu_usage_percent >", "HighCPUUsage"),
                "memory_threshold": ("system_memory_usage_percent >", "HighMemoryUsage"),
                "disk_threshold": ("system_disk_usage_percent >", "LowDiskSpace"),
                "error_rate_threshold": ("api_error_rate >", "HighAPIErrorRate"),
                "confidence_threshold": ("model_avg_confidence_score <", "LowModelConfidence"),
                "response_time_threshold": (") >", "HighResponseTime")
            }
            
            for threshold_key, threshold_value in thresholds.items():
                if threshold_key in threshold_mappings:
                    expr_pattern, alert_name = threshold_mappings[threshold_key]
                    rules_content = self._update_threshold_in_rules(
                        rules_content, expr_pattern, threshold_value, alert_name
                    )
            
            # Write updated rules
            with open(self.alert_rules_path, 'w') as f:
                f.write(rules_content)
            
            logger.info(f"Updated alert thresholds: {thresholds}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update alert thresholds: {e}")
            return False
    
    def _update_threshold_in_rules(self, content: str, expr_pattern: str, new_value: float, alert_name: str):
        """Helper method to update threshold in alert rules."""
        lines = content.split('\n')
        in_alert = False
        
        for i, line in enumerate(lines):
            if f"alert: {alert_name}" in line:
                in_alert = True
            elif in_alert and "expr:" in line and expr_pattern in line:
                # Update the threshold value
                if "<" in expr_pattern:
                    # For less-than comparisons (like confidence)
                    new_line = line.split('<')[0] + f"< {new_value}"
                else:
                    # For greater-than comparisons
                    new_line = line.split('>')[0] + f"> {new_value}"
                lines[i] = new_line
                in_alert = False
        
        return '\n'.join(lines)
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current alert configuration."""
        try:
            config = {
                "recipients": self.default_recipients,
                "thresholds": {
                    "cpu_threshold": 80,
                    "memory_threshold": 70,
                    "disk_threshold": 50,
                    "error_rate_threshold": 50,
                    "confidence_threshold": 0.6,
                    "response_time_threshold": 2
                },
                "last_updated": datetime.now().isoformat()
            }
            return config
        except Exception as e:
            logger.error(f"Failed to get current config: {e}")
            return {}
    
    def test_alert_config(self) -> Dict[str, bool]:
        """Test alert configuration files."""
        tests = {
            "alertmanager_config_exists": self.alertmanager_config_path.exists(),
            "alert_rules_exist": self.alert_rules_path.exists(),
            "config_readable": False,
            "rules_readable": False
        }
        
        try:
            with open(self.alertmanager_config_path, 'r') as f:
                f.read()
            tests["config_readable"] = True
        except Exception:
            pass
        
        try:
            with open(self.alert_rules_path, 'r') as f:
                f.read()
            tests["rules_readable"] = True
        except Exception:
            pass
        
        return tests 