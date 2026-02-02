#!/usr/bin/env python3
import socket
import yaml
import sys

def is_port_in_use(port, host='localhost'):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True

def find_free_port(start_port=10000, end_port=65535, host='localhost'):
    """Find an unused port within the specified range"""
    for port in range(start_port, end_port + 1):
        if not is_port_in_use(port, host):
            return port
    return None

def update_yaml_port(yaml_file, new_port=None):
    """Read YAML file, update port, and save"""
    try:
        # Read YAML file
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get current port
        current_port = config.get('main_process_port')
        print(f"Current port: {current_port}")
        
        # Check if current port is in use
        if current_port and is_port_in_use(current_port):
            print(f"Port {current_port} is in use, searching for available port...")
            
            # If no new port is specified, find one automatically
            if new_port is None:
                new_port = find_free_port(start_port=current_port + 1)
            
            if new_port:
                config['main_process_port'] = new_port
                print(f"Found available port: {new_port}")
                
                # Save updated YAML file
                with open(yaml_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"Updated configuration file: {yaml_file}")
                return new_port
            else:
                print("No available port found!")
                return None
        else:
            print(f"Port {current_port} is available")
            return current_port
            
    except FileNotFoundError:
        print(f"Error: File {yaml_file} does not exist")
        return None
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
        return None

def read_yaml_config(yaml_file):
    """Read and display YAML configuration"""
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        print("Current configuration:")
        print(yaml.dump(config, default_flow_style=False))
        return config
    except Exception as e:
        print(f"Failed to read configuration: {e}")
        return None

if __name__ == "__main__":
    # YAML configuration file path
    yaml_file = "util/deepspeed_configs/accelerate_config_single_node.yaml"
    
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    
    print(f"Checking configuration file: {yaml_file}\n")
    
    # Read current configuration
    read_yaml_config(yaml_file)
    
    print("\n" + "="*50 + "\n")
    
    # Check and update port
    free_port = update_yaml_port(yaml_file)
    
    if free_port:
        print(f"\nFinal port used: {free_port}")