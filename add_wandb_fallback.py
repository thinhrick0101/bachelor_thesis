#!/usr/bin/env python3
"""
add_wandb_fallback.py - Add WandB authentication fallback to training scripts
This script modifies training scripts to handle WandB authentication errors gracefully.
"""

import os
import re
import shutil
from pathlib import Path

def add_wandb_fallback_to_script(script_path):
    """Add WandB error handling to a training script."""
    
    print(f"📝 Processing {script_path}...")
    
    # Read the original script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Backup original
    backup_path = f"{script_path}.backup"
    shutil.copy(script_path, backup_path)
    print(f"💾 Backup saved to {backup_path}")
    
    # Define the WandB fallback code
    wandb_fallback_code = '''
# Enhanced WandB setup with authentication fallback
def setup_wandb_with_fallback(args):
    """Setup WandB with authentication fallback."""
    import os
    import wandb
    from wandb.errors import CommError, AuthenticationError
    
    print("🔑 Setting up WandB authentication...")
    
    # Try to initialize WandB with error handling
    try:
        # Force re-login to ensure authentication works
        wandb.login(relogin=True)
        
        # Initialize WandB run
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        print("✅ WandB initialized successfully")
        return True
        
    except (CommError, AuthenticationError) as e:
        print(f"⚠️ WandB authentication failed: {e}")
        print("🔄 Switching to offline mode...")
        
        # Set offline mode
        os.environ['WANDB_MODE'] = 'offline'
        
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                mode='offline'
            )
            print("✅ WandB initialized in offline mode")
            return False
            
        except Exception as offline_error:
            print(f"❌ Failed to initialize WandB even in offline mode: {offline_error}")
            print("📊 Training will continue without WandB logging")
            return False
    
    except Exception as e:
        print(f"❌ Unexpected WandB error: {e}")
        print("📊 Training will continue without WandB logging")
        return False

def safe_wandb_log(data):
    """Safely log to WandB with error handling."""
    try:
        if wandb.run is not None:
            wandb.log(data)
    except Exception as e:
        print(f"⚠️ WandB logging failed: {e}")
        # Continue training without logging
        pass
'''
    
    # Find where to insert the fallback code (after imports, before main)
    # Look for the main function or the wandb.init call
    lines = content.split('\n')
    insert_line = 0
    
    # Find a good place to insert (after imports, before main logic)
    for i, line in enumerate(lines):
        if 'def main(' in line or 'if __name__ == "__main__"' in line:
            insert_line = i
            break
        elif 'wandb.init(' in line:
            insert_line = i
            break
    
    if insert_line == 0:
        # Fallback: insert after imports
        for i, line in enumerate(lines):
            if line.strip() and not (line.startswith('import ') or line.startswith('from ') or line.startswith('#')):
                insert_line = i
                break
    
    # Insert the fallback code
    lines.insert(insert_line, wandb_fallback_code)
    
    # Replace direct wandb.init calls with the fallback function
    updated_content = '\n'.join(lines)
    
    # Replace wandb.init patterns
    wandb_init_pattern = r'wandb\.init\s*\(\s*([^)]+)\s*\)'
    def replace_wandb_init(match):
        return f"setup_wandb_with_fallback(args)  # Original: wandb.init({match.group(1)})"
    
    updated_content = re.sub(wandb_init_pattern, replace_wandb_init, updated_content)
    
    # Replace wandb.log calls with safe logging
    wandb_log_pattern = r'wandb\.log\s*\(\s*([^)]+)\s*\)'
    def replace_wandb_log(match):
        return f"safe_wandb_log({match.group(1)})"
    
    updated_content = re.sub(wandb_log_pattern, replace_wandb_log, updated_content)
    
    # Write the updated script
    with open(script_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Updated {script_path} with WandB fallback")
    return True

def main():
    """Main function to add WandB fallback to training scripts."""
    
    print("🔧 Adding WandB Authentication Fallback to Training Scripts")
    print("=" * 60)
    
    # Find training scripts
    training_scripts = [
        'bachelor_thesis/train_dense_model.py',
        'bachelor_thesis/train_sparse_transformer.py'
    ]
    
    success_count = 0
    
    for script in training_scripts:
        if os.path.exists(script):
            try:
                add_wandb_fallback_to_script(script)
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to process {script}: {e}")
        else:
            print(f"⚠️ Script not found: {script}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Successfully updated {success_count}/{len(training_scripts)} scripts")
    print("\n📋 Next steps:")
    print("1. Run: bash bachelor_thesis/fix_wandb_auth.sh")
    print("2. Run: bash bachelor_thesis/launch_experiments_parallel_fixed.sh")
    print("3. Your training will now gracefully handle WandB authentication errors")
    print("\n💡 If you need to restore original scripts, use the .backup files")

if __name__ == "__main__":
    main() 