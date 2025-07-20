# backend/bot/compile_cython.py
import subprocess
import sys
import os
from pathlib import Path


def compile_cython_extensions():
    """Compile Cython extensions with error handling"""

    cython_dir = Path("src/cython_extensions")

    if not cython_dir.exists():
        print("❌ Cython extensions directory not found")
        return False

    original_dir = os.getcwd()

    try:
        os.chdir(cython_dir)
        print("🔨 Compiling Cython extensions...")

        result = subprocess.run([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Cython extensions compiled successfully")
            print("📁 Generated files:")
            for file in cython_dir.glob("*.so"):
                print(f"   - {file.name}")
            for file in cython_dir.glob("*.pyd"):
                print(f"   - {file.name}")
            return True
        else:
            print("❌ Compilation failed:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    success = compile_cython_extensions()
    if success:
        print("\n🚀 Ready to test Cython acceleration!")
        print("Run your blueprint trainer to see the speedup.")
    else:
        print("\n⚠️ Compilation failed. Check error messages above.")
