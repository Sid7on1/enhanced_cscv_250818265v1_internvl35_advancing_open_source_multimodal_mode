import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'default_agent',
        'version': '1.0.0'
    },
    'environment': {
        'name': 'default_environment',
        'version': '1.0.0'
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(levelname)s - %(message)s'
    }
}

# Define data classes for configuration
@dataclass
class AgentConfig:
    name: str
    version: str

@dataclass
class EnvironmentConfig:
    name: str
    version: str

@dataclass
class LoggingConfig:
    level: str
    format: str

@dataclass
class Config:
    agent: AgentConfig
    environment: EnvironmentConfig
    logging: LoggingConfig

# Define exception classes
class ConfigError(Exception):
    pass

class ConfigLoadError(ConfigError):
    pass

class ConfigSaveError(ConfigError):
    pass

# Define configuration manager class
class ConfigurationManager:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Config:
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                return Config(
                    agent=AgentConfig(**config_data['agent']),
                    environment=EnvironmentConfig(**config_data['environment']),
                    logging=LoggingConfig(**config_data['logging'])
                )
        except FileNotFoundError:
            logger.warning(f'Config file not found: {self.config_file}')
            return Config(**DEFAULT_CONFIG)
        except yaml.YAMLError as e:
            raise ConfigLoadError(f'Failed to load config: {e}')

    def save_config(self, config: Config) -> None:
        try:
            config_data = {
                'agent': dict(config.agent),
                'environment': dict(config.environment),
                'logging': dict(config.logging)
            }
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        except Exception as e:
            raise ConfigSaveError(f'Failed to save config: {e}')

    def update_config(self, config: Config) -> None:
        self.config = config
        self.save_config(config)

    def get_config(self) -> Config:
        return self.config

# Define configuration class
class ConfigClass:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_config()

    def get_agent_config(self) -> AgentConfig:
        return self.config.agent

    def get_environment_config(self) -> EnvironmentConfig:
        return self.config.environment

    def get_logging_config(self) -> LoggingConfig:
        return self.config.logging

    def update_agent_config(self, name: str, version: str) -> None:
        self.config.agent = AgentConfig(name, version)
        self.config_manager.update_config(self.config)

    def update_environment_config(self, name: str, version: str) -> None:
        self.config.environment = EnvironmentConfig(name, version)
        self.config_manager.update_config(self.config)

    def update_logging_config(self, level: str, format: str) -> None:
        self.config.logging = LoggingConfig(level, format)
        self.config_manager.update_config(self.config)

# Create configuration manager and configuration class
config_manager = ConfigurationManager()
config = ConfigClass(config_manager)

# Example usage
if __name__ == '__main__':
    logger.info(f'Agent config: {config.get_agent_config()}')
    logger.info(f'Environment config: {config.get_environment_config()}')
    logger.info(f'Logging config: {config.get_logging_config()}')
    config.update_agent_config('new_agent', '1.1.0')
    config.update_environment_config('new_environment', '1.1.0')
    config.update_logging_config('DEBUG', '%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f'Updated agent config: {config.get_agent_config()}')
    logger.info(f'Updated environment config: {config.get_environment_config()}')
    logger.info(f'Updated logging config: {config.get_logging_config()}')