# Configuration file functions
def get_config_path():
    """Get path to configuration file"""
    return os.path.join(os.path.expanduser("~"), ".monitor_training_config.json")

def load_config():
    """Load configuration from file"""
    config_path = get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Config file {config_path} is invalid. Using defaults.")
    return {}

def save_config(config):
    """Save configuration to file"""
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False
