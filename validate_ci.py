#!/usr/bin/env python3
"""
CI/CD Validation Script
Validates that all components are working correctly before CI runs
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and return success status"""
    try:
        logger.info(f"🧪 Testing: {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"✅ {description} - PASSED")
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"💥 {description} - ERROR: {e}")
        return False

def validate_environment():
    """Validate CI/CD environment requirements"""
    
    logger.info("🚀 Starting CI/CD validation...")
    
    checks = [
        ("python3 --version", "Python 3 availability"),
        ("python3 -c 'import pandas, numpy, sklearn, fastapi, uvicorn, pytest'", "Core dependencies"),
        ("python3 create_sample_data.py", "Sample data creation"),
        ("python3 create_test_models.py", "Test model creation"),
        ("ls -la models/ | grep -E '(stage1|stage2|preprocessor)'", "Model files exist"),
        ("python3 -c 'import sys; sys.path.append(\"src\"); from serve.app import app; print(\"API imports OK\")'", "API imports"),
        ("python3 -m pytest tests/test_api.py::TestAPI::test_health_endpoint -v", "API health test"),
        ("python3 -m pytest tests/test_model_training.py::TestModelTraining::test_stage1_model_training_produces_file -v", "Model training test"),
        ("python3 -m pytest tests/test_feature_pipeline.py --collect-only", "Feature pipeline tests available"),
    ]
    
    passed = 0
    failed = 0
    
    for cmd, desc in checks:
        if run_command(cmd, desc):
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n📊 Validation Results:")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 All validation checks passed! CI/CD pipeline should work correctly.")
        return True
    else:
        logger.warning(f"⚠️  {failed} checks failed. CI/CD pipeline may encounter issues.")
        return False

if __name__ == '__main__':
    success = validate_environment()
    sys.exit(0 if success else 1)