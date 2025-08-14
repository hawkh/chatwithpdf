#!/usr/bin/env python3
"""
Migration script for PDF Chat Application
Helps users transition from the old version to the optimized version
"""
import os
import shutil
import json
import sys
from pathlib import Path

def print_banner():
    """Print migration banner"""
    print("=" * 60)
    print("🤖 PDF Chat Application - Migration Tool")
    print("=" * 60)
    print("This script will help you migrate from the old version")
    print("to the optimized version of the PDF Chat Application.")
    print("=" * 60)

def check_old_files():
    """Check for old application files"""
    old_files = [
        "main.py",
        "chat_with_pdf.py",
        "requirements.txt"
    ]
    
    found_files = []
    for file in old_files:
        if os.path.exists(file):
            found_files.append(file)
    
    return found_files

def backup_old_files():
    """Create backup of old files"""
    backup_dir = "backup_old_version"
    
    if os.path.exists(backup_dir):
        print(f"⚠️  Backup directory {backup_dir} already exists")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return False
    
    try:
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy old files
        old_files = ["main.py", "chat_with_pdf.py", "requirements.txt"]
        for file in old_files:
            if os.path.exists(file):
                shutil.copy2(file, backup_dir)
                print(f"✅ Backed up {file}")
        
        print(f"✅ Backup created in {backup_dir}/")
        return True
        
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False

def check_environment():
    """Check environment setup"""
    print("\n🔍 Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for .env file
    if os.path.exists(".env"):
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found - you'll need to create one")
    
    # Check for Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("✅ Google API key found in environment")
    else:
        print("⚠️  Google API key not found - you'll need to set it")
    
    return True

def install_dependencies():
    """Install new dependencies"""
    print("\n📦 Installing new dependencies...")
    
    try:
        # Install requirements
        os.system("pip install -r requirements.txt")
        print("✅ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_env_template():
    """Create .env template if it doesn't exist"""
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    env_template = """# PDF Chat Application - Environment Configuration

# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional - Vector Database Configuration
VECTOR_DB_TYPE=chroma  # Options: chroma, faiss, pinecone, qdrant
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# Optional - Caching Configuration
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600
MAX_CACHE_SIZE=1000

# Optional - Application Configuration
DEBUG=False
LOG_LEVEL=INFO
MAX_FILE_SIZE=52428800  # 50MB in bytes
"""
    
    try:
        with open(".env.template", "w") as f:
            f.write(env_template)
        print("✅ Created .env.template file")
        print("📝 Please copy .env.template to .env and configure your settings")
        return True
    except Exception as e:
        print(f"❌ Error creating .env template: {e}")
        return False

def migrate_data():
    """Migrate existing data if any"""
    print("\n📊 Checking for existing data...")
    
    # Check for old FAISS index
    if os.path.exists("faiss_index"):
        print("⚠️  Found old FAISS index")
        response = input("Do you want to migrate the old index? (y/N): ")
        if response.lower() == 'y':
            try:
                # Create data directory
                os.makedirs("data/vector_db", exist_ok=True)
                
                # Move old index
                shutil.move("faiss_index", "data/vector_db/faiss_index_old")
                print("✅ Old FAISS index moved to data/vector_db/faiss_index_old")
                print("📝 Note: You may need to reprocess your documents")
            except Exception as e:
                print(f"❌ Error migrating FAISS index: {e}")
    
    # Check for other data files
    data_files = [".cache", "logs", "temp"]
    for file in data_files:
        if os.path.exists(file):
            print(f"⚠️  Found {file} directory")
            response = input(f"Do you want to backup {file}? (y/N): ")
            if response.lower() == 'y':
                try:
                    backup_dir = f"backup_old_version/{file}"
                    shutil.move(file, backup_dir)
                    print(f"✅ {file} moved to {backup_dir}")
                except Exception as e:
                    print(f"❌ Error backing up {file}: {e}")

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data",
        "data/vector_db",
        "data/cache",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created {directory}/")
        except Exception as e:
            print(f"❌ Error creating {directory}: {e}")

def show_migration_summary():
    """Show migration summary"""
    print("\n" + "=" * 60)
    print("🎉 Migration Summary")
    print("=" * 60)
    print("✅ Old files backed up")
    print("✅ New dependencies installed")
    print("✅ Directory structure created")
    print("✅ Environment template created")
    print("\n📋 Next Steps:")
    print("1. Configure your .env file with your API keys")
    print("2. Run: streamlit run app.py")
    print("3. Upload and process your documents")
    print("\n📚 For more information, see README_OPTIMIZED.md")
    print("=" * 60)

def main():
    """Main migration function"""
    print_banner()
    
    # Check for old files
    old_files = check_old_files()
    if not old_files:
        print("❌ No old application files found")
        print("This migration script is for upgrading from the old version")
        return
    
    print(f"📁 Found old files: {', '.join(old_files)}")
    
    # Confirm migration
    response = input("\nDo you want to proceed with migration? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    # Start migration process
    print("\n🚀 Starting migration...")
    
    # Step 1: Backup old files
    if not backup_old_files():
        print("❌ Migration failed at backup step")
        return
    
    # Step 2: Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return
    
    # Step 4: Create .env template
    create_env_template()
    
    # Step 5: Create directories
    create_directories()
    
    # Step 6: Migrate data
    migrate_data()
    
    # Step 7: Show summary
    show_migration_summary()

if __name__ == "__main__":
    main()