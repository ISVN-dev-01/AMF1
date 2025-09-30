#!/usr/bin/env python3
"""
CI Failure Diagnosis Script
Simulates common CI failures locally to help debug GitHub Actions issues
"""

import subprocess
import sys
import os
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_dependency_installation():
    """Test if problematic dependencies can install"""
    logger.info("🧪 Testing problematic dependencies...")
    
    problematic_deps = [
        'lightgbm==4.1.0',
        'xgboost==3.0.5', 
        'prometheus-client==0.19.0',
        'shap==0.48.0'
    ]
    
    for dep in problematic_deps:
        try:
            logger.info(f"Testing {dep}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--dry-run', dep
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"✅ {dep} - Installation would succeed")
            else:
                logger.error(f"❌ {dep} - Installation would fail")
                logger.error(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {dep} - Installation timeout")
        except Exception as e:
            logger.error(f"💥 {dep} - Error: {e}")

def test_model_creation():
    """Test if model creation scripts work"""
    logger.info("🤖 Testing model creation...")
    
    scripts = [
        ('create_sample_data.py', 'Sample data creation'),
        ('create_test_models.py', 'Test model creation')
    ]
    
    for script, desc in scripts:
        if Path(script).exists():
            try:
                result = subprocess.run([
                    sys.executable, script
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info(f"✅ {desc} - SUCCESS")
                else:
                    logger.error(f"❌ {desc} - FAILED")
                    logger.error(f"Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"⏰ {desc} - TIMEOUT")
            except Exception as e:
                logger.error(f"💥 {desc} - ERROR: {e}")
        else:
            logger.warning(f"⚠️  {script} not found")

def test_api_imports():
    """Test if API can be imported without errors"""
    logger.info("🚀 Testing API imports...")
    
    test_script = '''
import sys
sys.path.append("src")
try:
    from serve.app import app
    print("SUCCESS: API imports work")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
except Exception as e:
    print(f"OTHER_ERROR: {e}")
'''
    
    try:
        result = subprocess.run([
            sys.executable, '-c', test_script
        ], capture_output=True, text=True, timeout=30)
        
        if "SUCCESS" in result.stdout:
            logger.info("✅ API imports - SUCCESS")
        else:
            logger.error("❌ API imports - FAILED")
            logger.error(f"Output: {result.stdout}")
            logger.error(f"Error: {result.stderr}")
            
    except Exception as e:
        logger.error(f"💥 API import test failed: {e}")

def test_pytest_availability():
    """Test if pytest can discover and run tests"""
    logger.info("🧪 Testing pytest functionality...")
    
    test_commands = [
        (['python3', '-m', 'pytest', '--version'], 'Pytest version'),
        (['python3', '-m', 'pytest', 'tests/', '--collect-only'], 'Test discovery'),
        (['python3', '-m', 'pytest', 'tests/test_api.py::TestAPI::test_health_endpoint', '-v'], 'Single test execution')
    ]
    
    for cmd, desc in test_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"✅ {desc} - SUCCESS")
            else:
                logger.error(f"❌ {desc} - FAILED")
                logger.error(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {desc} - TIMEOUT")
        except Exception as e:
            logger.error(f"💥 {desc} - ERROR: {e}")

def test_system_requirements():
    """Test system-level requirements"""
    logger.info("🖥️  Testing system requirements...")
    
    system_checks = [
        (['python3', '--version'], 'Python 3 availability'),
        (['pip3', '--version'], 'Pip3 availability'),
        (['git', '--version'], 'Git availability'),
        (['curl', '--version'], 'Curl availability')
    ]
    
    for cmd, desc in system_checks:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                logger.info(f"✅ {desc} - {version}")
            else:
                logger.error(f"❌ {desc} - NOT AVAILABLE")
                
        except FileNotFoundError:
            logger.error(f"❌ {desc} - COMMAND NOT FOUND")
        except Exception as e:
            logger.error(f"💥 {desc} - ERROR: {e}")

def analyze_requirements_file():
    """Analyze requirements.txt for potential issues"""
    logger.info("📋 Analyzing requirements.txt...")
    
    requirements_file = Path('requirements.txt')
    if not requirements_file.exists():
        logger.error("❌ requirements.txt not found")
        return
    
    with open(requirements_file) as f:
        requirements = f.read().strip().split('\n')
    
    logger.info(f"📦 Found {len(requirements)} dependencies")
    
    # Check for potentially problematic packages
    problematic_patterns = [
        ('lightgbm', 'Requires C++ compiler'),
        ('xgboost', 'Requires C++ compiler'), 
        ('shap', 'Heavy dependency with C extensions'),
        ('fastf1', 'May have network dependencies'),
        ('prometheus', 'May require additional setup')
    ]
    
    for req in requirements:
        for pattern, issue in problematic_patterns:
            if pattern in req.lower():
                logger.warning(f"⚠️  {req} - {issue}")

def generate_ci_debug_report():
    """Generate a comprehensive CI debug report"""
    logger.info("📊 Generating CI debug report...")
    
    report = {
        'timestamp': str(subprocess.run(['date'], capture_output=True, text=True).stdout.strip()),
        'python_version': sys.version,
        'platform': sys.platform,
        'working_directory': str(Path.cwd()),
        'environment_variables': {k: v for k, v in os.environ.items() if 'PATH' in k or 'PYTHON' in k}
    }
    
    # Check file structure
    important_files = [
        'requirements.txt',
        '.github/workflows/ci.yml',
        'src/serve/app.py',
        'tests/test_api.py',
        'create_sample_data.py',
        'create_test_models.py'
    ]
    
    report['file_status'] = {}
    for file_path in important_files:
        path = Path(file_path)
        report['file_status'][file_path] = {
            'exists': path.exists(),
            'size': path.stat().st_size if path.exists() else 0
        }
    
    # Save report
    with open('ci_debug_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("✅ CI debug report saved to ci_debug_report.json")

def main():
    """Run all CI diagnosis tests"""
    logger.info("🚀 Starting CI Failure Diagnosis...")
    logger.info("=" * 60)
    
    test_system_requirements()
    logger.info("-" * 40)
    
    analyze_requirements_file()
    logger.info("-" * 40)
    
    test_dependency_installation()
    logger.info("-" * 40)
    
    test_model_creation()
    logger.info("-" * 40)
    
    test_api_imports()
    logger.info("-" * 40)
    
    test_pytest_availability()
    logger.info("-" * 40)
    
    generate_ci_debug_report()
    
    logger.info("=" * 60)
    logger.info("🎯 CI Diagnosis Complete!")
    logger.info("📋 Check ci_debug_report.json for detailed information")
    logger.info("📋 Use this information to troubleshoot GitHub Actions failures")

if __name__ == '__main__':
    main()